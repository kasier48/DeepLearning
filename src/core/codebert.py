import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class CodeSummarizer:
    """
    긴 코드 스니펫을 여러 청크(Chunk)로 나눈 뒤,
    부분 요약 → 최종 요약(메타 요약)을 수행하는 클래스 예시.
    """

    def __init__(
        self,
        model_name="Salesforce/codet5-base-multi-sum",
        max_input_length=512,
        summary_length=100,
        num_beams=4,
        device=None
    ):
        """
        :param model_name: 사용할 CodeT5 모델 이름
        :param max_input_length: 하나의 청크가 될 수 있는 최대 토큰 수
        :param summary_length: 생성 요약의 최대 토큰 수
        :param num_beams: Beam Search 시 사용할 빔(beam) 개수
        :param device: GPU/CPU 설정 (None이면 자동 설정)
        """
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.summary_length = summary_length
        self.num_beams = num_beams

        # 디바이스 설정
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # 모델, 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.to(self.device)

    def chunk_text_by_tokens(self, text, overlap=0):
        """
        긴 텍스트(코드)를 토큰화한 뒤, max_input_length 단위로 청크를 분할.
        :param text: 코드 문자열
        :param overlap: 청크 사이의 오버랩 토큰 수 (정보 손실을 줄이기 위함)
        :return: 청크별 토큰 리스트의 리스트
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []

        start = 0
        while start < len(tokens):
            end = start + self.max_input_length
            chunk_tokens = tokens[start:end]
            chunks.append(chunk_tokens)
            # 다음 청크 시작점은 (max_input_length - overlap)만큼 이동
            start += self.max_input_length - overlap
        return chunks

    def generate_summary_for_chunk(self, chunk_tokens, prefix="summarize: "):
        """
        특정 청크(토큰 리스트)에 대해 부분 요약을 생성.
        :param chunk_tokens: 토큰 리스트
        :param prefix: 'summarize: ' 등 태스크를 명시할 접두어
        :return: 부분 요약 문자열
        """
        # Special tokens 등 추가
        input_ids = torch.tensor([self.tokenizer.build_inputs_with_special_tokens(chunk_tokens)])
        input_ids = input_ids.to(self.device)

        # 요약 생성
        summary_ids = self.model.generate(
            input_ids,
            max_length=self.summary_length,
            num_beams=self.num_beams,
            early_stopping=True
        )
        summary_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary_text

    def chunked_code_summarization(self, code_snippet, overlap=0):
        """
        코드 스니펫을 청크 단위로 요약하고,
        각 부분 요약을 최종적으로 다시 요약(메타 요약)하여 최종 결과를 반환.
        :param code_snippet: 긴 코드 문자열
        :param overlap: 청크 간 오버랩
        :return: 최종 요약 문자열
        """
        # 1) 코드 청킹
        code_chunks = self.chunk_text_by_tokens(code_snippet, overlap=overlap)

        # 2) 각 청크 부분 요약
        partial_summaries = []
        for i, chunk_tokens in enumerate(code_chunks):
            chunk_summary = self.generate_summary_for_chunk(chunk_tokens)
            partial_summaries.append(chunk_summary)

        if len(partial_summaries) == 1:
            return partial_summaries[0]
        
        # 3) 부분 요약들을 합쳐 최종 요약(메타 요약)
        combined_text = " ".join(partial_summaries)
        prefix = "summarize: "
        inputs = self.tokenizer.encode_plus(
            prefix + combined_text,
            return_tensors="pt",
            max_length=self.max_input_length,
            truncation=True
        )
        summary_ids = self.model.generate(
            inputs["input_ids"].to(self.device),
            max_length=self.summary_length,
            num_beams=self.num_beams,
            early_stopping=True
        )
        final_summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return final_summary