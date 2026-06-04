import regex as re
import json
import base64

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merge: dict[tuple[int, int], int],
        special_tokens: dict[str, int],
        pattern: str = None,
        cache_maxsize: int = 1<<20,
    ):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.merge = merge
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        #  按 chunk 字节序列做 LRU 缓存——自然语言里 chunk 高度重复，命中率高
        self._cache : dict[bytes, list[int]] = {}
        self._cache_maxsize = cache_maxsize


    def _encode_chunk(self, chunk_bytes: bytes) -> list[int]:
        cache = self._cache.get(chunk_bytes)
        if cache is not None:
            return cache

        merge = self.merge # 局部变量，避免反复属性查找
        ids = list(chunk_bytes)
        while len(ids) >= 2:
            min_rank = None
            min_id = -1
            for i in range(len(ids) - 1):
                cur_pair = (ids[i], ids[i+1])
                cur_rank = merge.get(cur_pair)
                if cur_rank is not None and (min_rank is None or cur_rank < min_rank):
                    min_rank = cur_rank
                    min_id = i
            if min_id < 0:
                # 没有任何可合并的 pair
                break
            ids[min_id: min_id+2] = [min_rank]

        if len(self._cache) < self._cache_maxsize:
            self._cache[chunk_bytes] = ids
        return ids



    def _encode_ordinary(self, text: str) -> list[int]:
        out: list[int] = []
        text_chunks = self.compiled_pattern.findall(text)
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8", errors="ignore")
            out.extend(self._encode_chunk(chunk_bytes))
        return out

    def encode(self, text) -> list[int]:
        special = self.special_tokens
        # 用正则拆分文本，把 special_tokens 作为分隔符，并使用捕获组 () 保留它们。
        # 例如，如果 special 包含 "<|endoftext|>"，则 special_pattern 变为 "(<|endoftext|>)"
        # re.split 会返回一个列表，其中 special_tokens 以单独元素的形式出现在结果中。
        special_pattern = "(" + "|".join(re.escape(k) for k in special.keys()) + ")"
        chunks = re.split(special_pattern, text)

        all_token_ids = []
        for chunk in chunks:
            if chunk in self.special_tokens:
                all_token_ids.append(self.special_tokens[chunk])
            else:
                all_token_ids.extend(self._encode_ordinary(chunk))

        return all_token_ids


    def decode(self, ids):
        part_bytes = []
        for id in ids:
            if id in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[id].encode("utf-8"))
            elif id in self.vocab:
                part_bytes.append(self.vocab[id])
            else:
                raise ValueError(f"invalid token id: {id}")
        text_bytes = b"".join(part_bytes)
        return text_bytes.decode("utf-8", errors="ignore")

    # save & load
    def save(self, path):
        """保存 tokenizer 到 JSON 文件"""
        # 将 vocab 中的 bytes 转为 base64 字符串
        vocab_serial = []
        for i in range(max(self.vocab.keys()) + 1):
            b = self.vocab.get(i)
            if b is not None:
                vocab_serial.append(base64.b64encode(b).decode("ascii"))
            else:
                vocab_serial.append(None)
        merges_list = []
        for (p0, p1), new_id in self.merge.items():
            merges_list.append([p0, p1, new_id])
        data = {
            "version": 1,
            "pattern": self.pattern,
            "special_tokens": self.special_tokens,
            "vocab": vocab_serial,
            "merges": merges_list
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"tokenizer saved to {path}\n"
              f"vocab_size: {self.vocab_size}\n"
              f"pattern: {self.pattern}\n"
              f"special_tokens: {self.special_tokens}\n")

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = {}
        for idx, b64_str in enumerate(data["vocab"]):
            if b64_str is not None:
                vocab[idx] = base64.b64decode(b64_str.encode("ascii"))
        merges = {}
        for item in data["merges"]:
            p0, p1, new_id = item
            merges[(p0, p1)] = new_id

        print(f"tokenizer loaded from {path}")

        return cls(
            vocab=vocab,
            merge=merges,
            special_tokens=data["special_tokens"],
            pattern=data["pattern"]
        )


if __name__ == "__main__":
    # load 已保存的 tokenzier
    tokenizer = Tokenizer.load("test_tokenizer.json")
    # 测试编码
    text = "Hello, world! <|endoftext|> This is a test."
    ids = tokenizer.encode(text)
    print("Token IDs:", ids)
    # 测试解码
    decoded = tokenizer.decode(ids)
    print("Decoded text:", decoded)

    # 文本测试
    valtext = "Many common characters, including numerals, punctuation, and other symbols, are unified within the standard and are not treated as specific to any given writing system. Unicode encodes thousands of emoji, with the continued development thereof conducted by the Consortium as a part of the standard.[4] Moreover, the widespread adoption of Unicode was in large part responsible for the initial popularization of emoji outside of Japan. Unicode is ultimately capable of encoding more than 1.1 million characters."
    valtext2 = tokenizer.decode(tokenizer.encode(valtext))
    print(valtext2 == valtext)