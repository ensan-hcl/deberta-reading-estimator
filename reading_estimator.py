from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from transformers import pipeline
from pyknp import Juman  # JUMAN tokenizer を使用


class ReadingPredictor:
    def __init__(self, model_name, references):
        self.jumanpp = Juman()  # Jumanを初期化
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.eval()
        self.references = references
        # replace [MASK] with tokenizer.mask_token
        for key, values in self.references.items():
            for reading, texts in values.items():
                self.references[key][reading] = [
                    self._split_reference(text) for text in texts
                ]
        self.reference_logits = self._calculate_reference_logits()

    def _calculate_reference_logits(self):
        # reference_logitsの初期化
        reference_logits = {}
        for kanji, readings in self.references.items():
            reference_logits[kanji] = {}
            for reading, examples in readings.items():
                reference_logits[kanji][reading] = []
                for text in examples:
                    inputs = self.tokenizer(text, return_tensors="pt")
                    outputs = self.model(**inputs)
                    mask_token_index = torch.where(
                        inputs["input_ids"][0] == self.tokenizer.mask_token_id
                    )[0]
                    reference_logits[kanji][reading].append(
                        outputs.logits[0, mask_token_index]
                    )
        return reference_logits

    def _get_most_similar_token(self, kanji, logit):
        max_similarity = 0
        predicted_reading = None
        # 与えられた漢字に対する全ての読みを確認
        for reading, values in self.reference_logits[kanji].items():
            for value in values:
                similarity = torch.nn.functional.cosine_similarity(
                    logit, value, dim=1
                ).item()
                if similarity > max_similarity:
                    max_similarity = similarity
                    predicted_reading = reading
        return predicted_reading
    
    def _split_reference(self, text):
        # referenceのテキストを形態素解析し、半角スペースで分割する
        result = self.jumanpp.analysis(text)
        text = " ".join([mrph.midasi for mrph in result.mrph_list()])
        text = text.replace("[ MASK ]", self.tokenizer.mask_token)
        print(text)
        return text
    

    def get_reading_prediction(self, text):
        # Jumanでテキストを形態素解析し、分割する
        result = self.jumanpp.analysis(text)
        predicted_readings = {}

        for mrph in result.mrph_list():
            if mrph.midasi in self.references:  # 原形が対象の読み分け単語に含まれる場合
                masked_text = text.replace(mrph.midasi, self.tokenizer.mask_token)
                inputs = self.tokenizer(masked_text, return_tensors="pt")
                outputs = self.model(**inputs)
                mask_token_index = torch.where(
                    inputs["input_ids"][0] == self.tokenizer.mask_token_id
                )[0]
                predicted_reading = self._get_most_similar_token(
                    mrph.midasi, outputs.logits[0, mask_token_index]
                )
                predicted_readings[mrph.midasi] = predicted_reading

        return predicted_readings


if __name__ == "__main__":
    # 使用例
    references = {
        "金": {
            "かね": [
                f"私は[MASK]を稼ぎたいです",
                f"たくさんの[MASK]があればなんでも買えます",
            ],
            "きん": [
                f"[MASK]と銀の採掘を行う仕事",
                f"私は[MASK]メダルを取りたいです",
                f"先週の[MASK]の価格はどうでしたか",
                f"毎週[MASK]曜日に授業があります",
            ],
            "きむ": [
                f"[MASK]先生はとても厳しいです",
                f"韓国の[MASK]はとても有名です",
                f"[MASK]正恩は北朝鮮の最高指導者です",
            ],
        },
        "弾く": {
            "ひく": [
                f"昨日鍵盤ハーモニカを[MASK]経験をしました",
            ],
            "はじく": [
                f"表面で光を[MASK]ことを反射といいます",
            ],
        },
        "行った": {
            "いった": [
                f"昨日は公園に散歩に[MASK]",
            ],
            "おこなった": [
                f"この仮説を検証するため、実験を[MASK]"
            ],
        },
    }

    predictor = ReadingPredictor("ku-nlp/deberta-v2-base-japanese", references)

    texts = [
        "結局世の中は金が全てです",
        "王水は金も溶かす強力な溶液です",
        "金正日が来日した",
        "ピアノを弾くのが好きです",
        "ギターの弦を弾くと音が出ます",
        "油は水を弾く",
        "学校に行った",
        "開会式を行った",
    ]

    for text in texts:
        predicted_readings = predictor.get_reading_prediction(text)
        print(f"Original text: {text}")
        for word, reading in predicted_readings.items():
            print(f"Predicted reading for '{word}': {reading}")
