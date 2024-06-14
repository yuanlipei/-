import pickle
import collections
from nltk.tokenize import word_tokenize


# 定义自定义的NGramModel类
class NGramModel:
    def __init__(self, n, lambda_values):
        self.n = n
        self.lambda_values = lambda_values
        self.ngram_counts = collections.defaultdict(collections.Counter)
        self.context_counts = collections.Counter()
        self.vocab = set()

    def read_text_file(self, file_path):
        """读取文本文件，按行返回内容"""
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        return lines

    def build_ngram_model(self, lines):
        """建立n元语法模型"""
        for line in lines:
            words = word_tokenize(line.strip().lower())  # 使用word_tokenize分词
            if len(words) < self.n:
                continue
            # 遍历每个n元组合
            for i in range(len(words) - self.n + 1):
                ngram = tuple(words[i:i + self.n])
                context = ngram[:-1]  # 前n-1个词作为上下文
                target_word = ngram[-1]  # 最后一个词作为目标词

                self.ngram_counts[context][target_word] += 1
                self.context_counts[context] += 1
                self.vocab.add(target_word)

    def calculate_ngram_probability(self, context):
        """计算给定上下文的ngram概率分布"""
        if tuple(context) in self.ngram_counts:
            total_count = float(self.context_counts[tuple(context)])
            return {word: count / total_count for word, count in self.ngram_counts[tuple(context)].items()}
        else:
            return {}

    def linear_interpolation(self, context):
        """使用线性插值法计算平滑后的概率分布"""
        interpolation_prob = {}
        # 计算每个n元模型的概率分布
        for i in range(self.n):
            current_context = context[-(i + 1):]
            current_lambda = self.lambda_values[i]
            current_prob = self.calculate_ngram_probability(current_context)
            # 累加到插值概率分布中
            for word, prob in current_prob.items():
                interpolation_prob[word] = interpolation_prob.get(word, 0) + current_lambda * prob

        return interpolation_prob

    def predict_next_word(self, context, possible_words):
        """根据插值后的概率预测下一个词"""
        interpolated_prob = self.linear_interpolation(context)

        max_prob = 0
        next_word = None
        for word in possible_words:
            if word in interpolated_prob and interpolated_prob[word] > max_prob:
                max_prob = interpolated_prob[word]
                next_word = word
        return next_word

    def train(self, file_path):
        """训练n元语法模型"""
        lines = self.read_text_file(file_path)
        self.build_ngram_model(lines)

    def predict(self, context, possible_words):
        """预测最可能的下一个词"""
        return self.predict_next_word(context, possible_words)

    def save_model(self, file_path):
        """保存模型到文件"""
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(file_path):
        """从文件加载模型"""
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        return model
