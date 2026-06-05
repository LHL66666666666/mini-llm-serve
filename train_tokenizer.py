import heapq
from collections import Counter, defaultdict
import os
from typing import BinaryIO
from tqdm import tqdm
import time
import regex as re
# 多进程
import multiprocessing
from multiprocessing import Pool, cpu_count
# 可选内存监控支持
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    将一个大型文本文件按安全的位置切割成若干块，以便后续分块进行预分词和词频统计。
    绝不切断特殊标记（如 <|endoftext|>），从而保证每个块都可以独立地被正确解析，且不破坏标记的完整性。
    :param file: 已按二进制模式打开的语料文件（如 open("corpus.txt", "rb")）
    :param desired_num_chunks: 希望将文件分割成的块数。通常与并行进程数或内存控制相关
    :param split_special_token: 用于“对齐”分块的特殊标记，必须是字节串（如 b"<|endoftext|>"）。该标记会被当作安全的分割点，确保它被完整地放到某一个块的开头
    :return: 一个递增的整数列表，表示文件中各个块的字节偏移量。例如 [0, 1024, 2048, 4096] 表示有 3 个块（区间分别为 [0,1024), [1024,2048), [2048,4096)）
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))



# 预分词 pre tokenize + 在预分词阶段词频统计
def pre_tokenize_and_count(
    args: tuple[bytes, dict[str, int], re.Pattern, re.Pattern]
) -> Counter:
    """
    预分词 + 在预分词阶段词频统计
    :param args:
    chunk_bytes：该 chunk 的原始字节。
    special_token_to_id：特殊 token 到 id 的映射 , dict: str -> int 。
    delimiter_pattern_compiled：正则模式，用于识别特殊 token 作为分隔符，把它们切到独立的子块中。
    PAT_COMPILED：已编译的正则表达式对象，将原始文本预切分
    """
    chunk_bytes, special_token_to_id, delimiter_pattern_compiled, PAT_COMPILED = args
    # 解码为原来的字符串
    chunk = chunk_bytes.decode("utf-8", errors="ignore")
    # 建立 special token 集合
    special_tokens_set = set(special_token_to_id.keys())

    words_list = Counter()

    # split 特殊 token ，如果存在 special token ，将其单独拆出来，因为 special token 必须不可拆分
    if delimiter_pattern_compiled:
        sub_chunks = delimiter_pattern_compiled.split(chunk)
    else:
        sub_chunks = [chunk]

    for sub_chunk in sub_chunks:
        if not sub_chunk:
            continue

        if sub_chunk in special_tokens_set:
            special_token_id = special_token_to_id[sub_chunk]
            # 统一使用 tuple
            words_list[(special_token_id,)] += 1
        else:
            # PAT_COMPILED 是预设好的 GPT-2/GPT-4 的 regex
            for word in PAT_COMPILED.findall(sub_chunk):
                # 按照预设pattern将块切分为词
                id_sequence = tuple(word.encode("utf-8", errors="ignore"))
                words_list[id_sequence] += 1

    return words_list

# 假设文本在预分词后，我们有一个包含 n 个符号（字节）的序列 [A, B, C, D, ...]，要合并 m 次
# 原始BPE算法每轮合并都会扫描整个序列，统计所有相邻符号对 (A,B), (B,C) 的出现次数
# 找出频率最高且存在于词表的一对，比如 (B,C)。再次扫描整个序列，将所有的 B, C 替换为新的符号 BC
# 这个过程是 O(m × n) 复杂度。问题在于，每轮合并后的序列变化很小，但下一轮依然要全量重新计算所有符号对的频率，产生了巨大的重复计算
# 优化思路：双向链表 + pair索引 + 优先队列 + 惰性删除
# 使用优先队列优化：增量更新，而非全量重算。复杂度为 O(m log n)
# 核心数据结构有四个：
# 双向链表：维护每个word内部的token序列。每个节点代表一个符号，并指向其左右邻居。这使得合并操作（移除两个旧节点，插入一个新节点）的成本变为 O(1)，如果使用list，复杂度将变为O(n)
# pair_to_nodes：pair -> 所有出现位置
# pair_freqs：pair -> 总频率
# 优先队列（小顶堆）：按频率排序，存储所有符号对。堆顶永远是下一个要合并的、频率最高的符号对

# 优先队列中的元素定义 与 重载小于比较运算符 <
class pq_item():
    def __init__(self, freq, id_pair, byte_pair):
        # 这个pair(二元组)的出现频率
        self.freq = freq
        # pair
        self.id_pair = id_pair
        # pair对应的bytes对象
        self.byte_pair = byte_pair

    def __lt__(self, other):
        # __lt__ 返回true表示self"更小"
        # 先按照频率排序，频率相同取字典序较大
        if self.freq != other.freq:
            return self.freq > other.freq
        return self.byte_pair > other.byte_pair

# 双向链表中的元素定义
class Node():
    def __init__(self, token_id, word_freq):
        # value : 节点对应的 token_id
        self.value = token_id
        # word_freq : 一同一个 word 的所有 node 共享同一个频率引用
        # 例如：hello出现1000次，它的：h e l l o 所有 node共享：{'count':1000} 避免：每个node复制频率，节省大量内存。
        self.word_freq = word_freq
        # 前驱和后继
        self.prev = None
        self.next = None


# 训练bbpe
def train_bbpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    pattern_compiled: re.Pattern[str],
    num_chunks: int = 8,
    num_processes: int = None,
    **kwargs,
) -> tuple[dict[int, bytes], dict[tuple[int, int], int], dict[str, int], dict]:
    # -------------------预分词阶段-------------------------
    # 计时开始：before_pretokenization_time 用于记录预分词之前的准备工作耗时
    before_pretokenization_time = time.time()
    # vocab : int -> bytes 词汇表
    vocab = {idx: bytes([idx]) for idx in range(256)}
    # special_token_to_id : str -> int
    special_token_to_id = {}

    for special_token in special_tokens:
        special_token_bytes = special_token.encode("utf-8")
        if special_token_bytes not in vocab.values():  # 统一比较 bytes
            special_token_id = len(vocab)
            vocab[special_token_id] = special_token_bytes

            special_token_to_id[special_token] = special_token_id

    # byte_to_token_id : vocab 逆向版本 , bytes -> int
    byte_to_token_id = {v: k for k, v in vocab.items()}

    # 用于识别特殊 token 作为分隔符的正则模式
    delimiter_pattern_compiled = None
    if special_tokens:
        # 将特殊标记按长度降序排序，避免短标记匹配时干扰长标记（例如 <|endoftext|> 包含 <|end 等情况）
        special_tokens_sorted = sorted(
            [t.encode("utf-8") for t in special_tokens], key=len, reverse=True
        )
        escaped_tokens = [re.escape(t.decode("utf-8")) for t in special_tokens_sorted]
        delimiter_pattern = "|".join(escaped_tokens)
        if delimiter_pattern:
            delimiter_pattern_compiled = re.compile(f"({delimiter_pattern})")

    # 读文件
    with open(input_path, "rb") as f:
        # 把大文件切 chunk
        boundaries = find_chunk_boundaries(
            f, num_chunks, "<|endoftext|>".encode("utf-8")
        )

        chunk_args = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk_bytes = f.read(end - start)
            chunk_args.append(
                (
                    chunk_bytes,
                    special_token_to_id,
                    delimiter_pattern_compiled,
                    pattern_compiled,
                )
            )

    # # 串行处理
    # # 预分词准备工作耗时
    # before_pretokenization_time = time.time() - before_pretokenization_time
    # print(f"Time taken before pretokenization: {before_pretokenization_time:.2f} seconds")
    # # 预分词词频统计
    # all_word_freqs = Counter()
    # start_time = time.time()
    # for chunk_arg in tqdm(chunk_args, desc="Processing chunks"):
    #     chunk_counter = pre_tokenize_and_count(chunk_arg)
    #     all_word_freqs.update(chunk_counter)
    #
    # end_time = time.time()
    # print(f"Pre-tokenization and initial counting time: {end_time - start_time:.2f} seconds")

    # 改进为并行
    # 确定使用的进程数
    if num_processes is None:
        num_processes = min(cpu_count(), 8)  # 避免创建过多进程
    processes_to_use = min(num_processes, len(chunk_args))
    # 打印信息
    before_pretokenization_time = time.time() - before_pretokenization_time
    print(f"Time taken before pretokenization: {before_pretokenization_time:.2f} seconds")
    # 使用 spawn 上下文创建池
    ctx = multiprocessing.get_context("spawn")

    all_word_freqs = Counter()
    start_time = time.time()

    with ctx.Pool(processes=processes_to_use) as pool:
        print(f"Starting pre-tokenization with {processes_to_use} processes on {len(chunk_args)} chunks...")

        # imap_unordered 逐个返回结果，内存友好
        # 设置 chunksize 可以批量发送任务，提高效率（一般设为 max(1, len(chunk_args)//(processes_to_use*4))）
        chunksize = max(1, len(chunk_args) // (processes_to_use * 4))
        results_iter = pool.imap_unordered(
            pre_tokenize_and_count,
            chunk_args,
            chunksize=chunksize
        )

        # 合并每个 chunk 的词频
        for chunk_counter in tqdm(results_iter, total=len(chunk_args), desc="Processing chunks"):
            all_word_freqs.update(chunk_counter)
    end_time = time.time()
    print(f"Pre-tokenization and initial counting time: {end_time - start_time:.2f} seconds")
    # 记录观测指标：预分词时间
    pretokenize_duration = end_time - start_time



    # -------------------merge阶段-------------------------
    # 建立辅助数据结构
    # pair_to_nodes : dict{ tuple -> set } pair对应节点nodes,所有出现该pair的位置
    # 例如：语料：hello,help
    # pair：(h,e) 可能对应： {nodeA, nodeB, nodeC}
    # 这些 node都是：pair左边token的位置，即：node.value = h , node.next.value = e
    pair_to_nodes = defaultdict(set)
    # 此处迭代的word是一个tuple，count是word出现频率
    for word, count in all_word_freqs.items():
        if len(word) < 2:
            continue

        word_freq = {'count': count}
        prev_node = Node(word[0], word_freq)
        pair_to_nodes[(word[0], word[1])].add(prev_node)

        for i in range(1, len(word)):
            curr_node = Node(word[i], word_freq)
            if i+1 < len(word):
                pair_to_nodes[(word[i], word[i+1])].add(curr_node)

            curr_node.prev = prev_node
            prev_node.next = curr_node

            prev_node = curr_node

    # 记录每个pair出现频率，用于建立优先队列和实现优先队列的懒惰删除
    pair_freqs = Counter()
    for pair, nodes in pair_to_nodes.items():
        # nodes是一个集合，存储所有出现pair的节点，而每个节点有word_freq表示每个单词出现次数，因此每个节点相当于出现word_freq个pair
        pair_freqs[pair] = sum(node.word_freq['count'] for node in nodes)

    # 建立优先队列(原地建堆)
    pq =[pq_item(freq, pair, (vocab[pair[0]], vocab[pair[1]])) for pair, freq in pair_freqs.items()]
    heapq.heapify(pq)

    # 合并
    num_merges = vocab_size - len(vocab)
    pbar = tqdm(total=num_merges, desc="Performing BPE merges")

    # 记录观测指标：merge开始时的unique pair数量，heap大小
    initial_pair_count = len(pair_freqs)  # merge开始时的unique pair数量
    initial_heap_size = len(pq)  # 初始堆大小
    # 记录观测指标：最大堆大小和内存峰值
    max_heap_size = len(pq)
    peak_memory_mb = None
    if _HAS_PSUTIL:
        process = psutil.Process()
        # 记录初始内存
        peak_memory_mb = process.memory_info().rss / 1024 / 1024

    start_time = time.time()

    merges = {}

    for _ in range(num_merges):
        # 更新最大堆大小
        max_heap_size = max(max_heap_size, len(pq))
        top_pair = None
        while len(pq) > 0:
            # 优先队列中可能存在过期元素，过期元素跳过
            item = heapq.heappop(pq)
            # 元素不存在了(被del了)
            if item.id_pair not in pair_freqs:
                continue
            # 频率恰好等于pair_freqs的频率，表示不是过期元素
            if item.freq == pair_freqs[item.id_pair]:
                top_pair = item.id_pair
                break
        # 无可合并pair
        if not top_pair:
            break

        # 训练时监控 pair 的数量和堆的大小
        if _ % 1000 == 0:
            print(f"\nmerge {_}: unique pairs in pair_freqs: {len(pair_freqs)}, heap size: {len(pq)}")


        # 每1000步采样一次内存
        if _HAS_PSUTIL and _ % 1000 == 0:
            mem = process.memory_info().rss / 1024 / 1024
            if mem > peak_memory_mb:
                peak_memory_mb = mem

        p0, p1 = top_pair
        new_token_id = len(vocab)
        vocab[new_token_id] = vocab[p0] + vocab[p1]
        merges[(p0, p1)] = new_token_id

        # 待处理的节点转为list，因为不能一遍迭代一遍修改set
        nodes_to_process = list(pair_to_nodes[top_pair])
        for node in nodes_to_process:
            # 当语料中出现连续相同的字符（例如 A A A），且当前准备合并的 top_pair 恰好是 (A, A) 时，pair_to_nodes[(A, A)] 会同时包含第一个A节点和第二个A节点
            # 在修改第一个的时候，会同时改掉第二个，导致出现问题
            # 添加有效性校验：确保 node 及其后继节点没有在之前的循环中被修改
            if node.value != p0 or node.next is None or node.next.value != p1:
                continue

            # node一定存在后继节点，否则证明node在结尾处，与pair矛盾
            next_node = node.next

            # 每个词共用一个word_freq
            word_freq = node.word_freq['count']

            # 四个节点受到影响，分别更新：prev  cur  next  next.next
            if node.prev is not None:
                prev_node = node.prev
                # 更新频率(pair_to_nodes,pair_freqs,pq 三者同步更新)
                # 删除旧的 旧的统一以old_为命名前缀
                old_prev_pair = (prev_node.value, node.value)
                pair_to_nodes[old_prev_pair].discard(prev_node)
                pair_freqs[old_prev_pair] -= word_freq
                heapq.heappush(pq, pq_item(pair_freqs[old_prev_pair], old_prev_pair,
                                           (vocab[old_prev_pair[0]], vocab[old_prev_pair[1]])))

                old_curr_pair = (node.value, next_node.value)
                pair_to_nodes[old_curr_pair].discard(node)
                pair_freqs[old_curr_pair] -= word_freq # top_pair，结尾 del，不需 push 到 heapq 中

                if next_node.next is not None:
                    old_next_pair = (next_node.value, next_node.next.value)
                    pair_to_nodes[old_next_pair].discard(next_node)
                    pair_freqs[old_next_pair] -= word_freq
                    heapq.heappush(pq, pq_item(pair_freqs[old_next_pair], old_next_pair,
                                               (vocab[old_next_pair[0]], vocab[old_next_pair[1]])))

                # 加入新的 新的统一以new_为命名前缀
                new_prev_pair = (prev_node.value, new_token_id)
                pair_to_nodes[new_prev_pair].add(prev_node)
                pair_freqs[new_prev_pair] += word_freq
                heapq.heappush(pq, pq_item(pair_freqs[new_prev_pair], new_prev_pair, (vocab[new_prev_pair[0]], vocab[new_prev_pair[1]])))

                if next_node.next is not None:
                    new_curr_pair = (new_token_id, next_node.next.value)
                    pair_to_nodes[new_curr_pair].add(next_node)
                    pair_freqs[new_curr_pair] += word_freq
                    heapq.heappush(pq, pq_item(pair_freqs[new_curr_pair], new_curr_pair, (vocab[new_curr_pair[0]], vocab[new_curr_pair[1]])))

                # 处理前驱后继节点
                prev_node.next = next_node
                next_node.prev = node.prev

            else:
                # 更新频率
                # 删除旧的
                old_curr_pair = (node.value, next_node.value)
                pair_to_nodes[old_curr_pair].discard(node)
                pair_freqs[old_curr_pair] -= word_freq

                if next_node.next is not None:
                    old_next_pair = (next_node.value, next_node.next.value)
                    pair_to_nodes[old_next_pair].discard(next_node)
                    pair_freqs[old_next_pair] -= word_freq
                    heapq.heappush(pq, pq_item(pair_freqs[old_next_pair], old_next_pair,
                                               (vocab[old_next_pair[0]], vocab[old_next_pair[1]])))


                # 加入新的
                if next_node.next is not None:
                    new_curr_pair = (new_token_id, next_node.next.value)
                    pair_to_nodes[new_curr_pair].add(next_node)
                    pair_freqs[new_curr_pair] += word_freq
                    heapq.heappush(pq, pq_item(pair_freqs[new_curr_pair], new_curr_pair, (vocab[new_curr_pair[0]], vocab[new_curr_pair[1]])))

                # 处理前驱后继节点
                next_node.prev = None


            next_node.value = new_token_id

        del(pair_to_nodes[top_pair])
        del(pair_freqs[top_pair])

        pbar.update(1)

    end_time = time.time()
    print(f"Merge time: {end_time - start_time:.2f} seconds")
    merge_duration = end_time - start_time
    final_pair_count = len(pair_freqs)
    final_heap_size = len(pq)
    # 最后采样一次内存
    if _HAS_PSUTIL:
        mem = process.memory_info().rss / 1024 / 1024
        if mem > peak_memory_mb:
            peak_memory_mb = mem
    # 构建stats字典
    stats = {
        "pretokenize_time_sec": round(pretokenize_duration, 2),
        "merge_time_sec": round(merge_duration, 2),
        "total_time_sec": round(pretokenize_duration + merge_duration, 2),
        "initial_pair_count": initial_pair_count,
        "final_pair_count": final_pair_count,
        "initial_heap_size": initial_heap_size,
        "max_heap_size": max_heap_size,
        "final_heap_size": final_heap_size,
        "peak_memory_mb": round(peak_memory_mb, 2) if peak_memory_mb is not None else None,
    }

    pbar.close()
    return vocab, merges, special_token_to_id, stats


def train_tokenizer(
        filepath,
        vocab_size,
        special_tokens,
        num_chunks=32,
        num_processes=8,
):
    GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    PATTERN_CHOSEN = GPT4_SPLIT_PATTERN
    PAT_COMPILED = re.compile(PATTERN_CHOSEN)
    vocab, merges, special_token_to_id, stats = train_bbpe(
        input_path=filepath,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        pattern_compiled=PAT_COMPILED,
        num_chunks=num_chunks,
        num_processes=num_processes,
    )
    return vocab, merges, special_token_to_id, PATTERN_CHOSEN, stats

if __name__ == "__main__":
    # cl100k_base : ["<|endoftext|>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|endofprompt|>"]
    special_tokens = ["<|endoftext|>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|endofprompt|>"]
    vocab, merges, special_token_to_id, PATTERN_CHOSEN, stats = (
        train_tokenizer(filepath="D:/FineWeb-edu_data/tokenizer_heldout.txt", vocab_size=50304, special_tokens=special_tokens)
    )
    from tokenizer import Tokenizer
    tokenizer = Tokenizer(
        vocab=vocab,
        merge=merges,
        special_tokens=special_token_to_id,
        pattern=PATTERN_CHOSEN
    )
    tokenizer.save("test_tokenizer.json")
    # print(vocab)
    # print("=======================================================")
    # print(merges)
    # print("=======================================================")
    # print(special_token_to_id)
    # print("=======================================================")

