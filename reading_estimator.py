from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from transformers import pipeline
from pyknp import Juman  # JUMAN tokenizer を使用
import json
from copy import deepcopy

class ReadingEstimator:
    def __init__(self, model_name, references, evaluation_type="most_similar"):
        """
        Args:
            model_name (str): 使用するモデルの名前
            references (dict): 参照データ
            evaluation_type (str): 評価方法
             - most_similar: コサイン類似度が最も高い読みを予測
             - average: すべての参照データのコサイン類似度の平均が最も高い読みを予測
        """
        self.jumanpp = Juman()  # Jumanを初期化
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.eval()
        self.references = deepcopy(references)
        self.evaluation_type = evaluation_type
        # replace [MASK] with tokenizer.mask_token
        for key, values in self.references.items():
            for reading, texts in values.items():
                self.references[key][reading] = [
                    self._split_reference(text) for text in texts
                ]
        self.reference_logits = self._calculate_reference_logits()

    def update_references(self, references):
        self.references = deepcopy(references)
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
                        (outputs.logits[0, mask_token_index], text)
                    )
        return reference_logits

    def _get_most_similar_reading(self, kanji, logit):
        max_similarity = 0
        predicted_reading = None
        # 与えられた漢字に対する全ての読みを確認
        for reading, values in self.reference_logits[kanji].items():
            for (value, text) in values:
                similarity = torch.nn.functional.cosine_similarity(
                    logit, value, dim=1
                ).item()
                # print(f"{reading}, {similarity:04f}, {text}")
                if similarity > max_similarity:
                    max_similarity = similarity
                    predicted_reading = reading
        return predicted_reading
    
    def _get_average_similar_reading(self, kanji, logit):
        max_similarity = 0
        predicted_reading = None
        # 与えられた漢字に対する全ての読みを確認
        for reading, values in self.reference_logits[kanji].items():
            similarity_sum = 0
            for (value, text) in values:
                similarity_sum += torch.nn.functional.cosine_similarity(
                    logit, value, dim=1
                ).item()
            similarity = similarity_sum / len(values)
            if similarity > max_similarity:
                max_similarity = similarity
                predicted_reading = reading
        return predicted_reading

    def _split_reference(self, text):
        # referenceのテキストを形態素解析し、半角スペースで分割する
        result = self.jumanpp.analysis(text)
        text = " ".join([mrph.midasi for mrph in result.mrph_list()])
        text = text.replace("[ MASK ]", self.tokenizer.mask_token)
        return text

    def get_reading_prediction(self, text):
        # Jumanでテキストを形態素解析し、分割する
        result = self.jumanpp.analysis(text)
        predicted_readings = []

        for mrph in result.mrph_list():
            # FIXME: 一文に複数回出現する場合に対応
            if mrph.midasi in self.references and text.count(mrph.midasi) == 1:  # 原形が対象の読み分け単語に含まれる場合
                masked_text = " ".join([
                    self.tokenizer.mask_token if mrph.midasi == item.midasi else item.midasi for item in result.mrph_list()
                ])
                inputs = self.tokenizer(masked_text, return_tensors="pt")
                outputs = self.model(**inputs)
                mask_token_index = torch.where(
                    inputs["input_ids"][0] == self.tokenizer.mask_token_id
                )[0]
                get_reading = self._get_most_similar_reading if self.evaluation_type == "most_similar" else self._get_average_similar_reading
                predicted_reading = get_reading(
                    mrph.midasi, outputs.logits[0, mask_token_index]
                )
                predicted_readings.append((mrph.midasi, predicted_reading))
            else:
                predicted_readings.append((mrph.midasi, mrph.yomi))
        return predicted_readings


if __name__ == "__main__":
    # 使用例
    references = json.load(open("references.json", "r"))
    # 「水」以外のkeyを削除
    # references = {key: references[key] for key in references if key == "水"}
    predictor = ReadingEstimator("ku-nlp/deberta-v2-base-japanese", references, evaluation_type="most_similar")

    texts = [
        "結局世の中は金が全てです",
        "王水は金も溶かす強力な溶液です",
        "金正日が来日した",
        "ピアノを弾くのが好きです",
        "ギターの弦を弾くと音が出ます",
        "油は水を弾く",
        "生理食塩水",
        "学校に行った",
        "開会式を行った",
        "君と僕の間で何か隠し事があるのは良くない",
        "紅葉が綺麗に色づく季節になりました",
        "紅葉した山の景色",
        "北の方に向かって進む",
        "例のあの方がいらっしゃいました",
        "その件については私に任せてください",
        "件の人物を探し出す",
        "集合の元aに対して、-aが常に存在する",
        "彼の元には多くの人が集まった",
    ]

    for text in texts:
        predicted_readings = predictor.get_reading_prediction(text)
        print(f"Original text: {text}")
        joined_yomi = "".join([yomi for _, yomi in predicted_readings])
        print(f"Predicted readings: {joined_yomi}")
