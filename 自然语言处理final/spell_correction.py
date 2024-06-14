import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import reuters
import subprocess
from own_model import NGramModel

# 下载必要的 NLTK 资源
nltk.download('reuters')
nltk.download('punkt')

# 加载词典
def load_dictionary(vocab_path):
    with open(vocab_path, 'r') as file:
        vocab = set(file.read().split())
    return vocab

def exchange_letter(word):
    # 转换为列表以便于交换字母
    word_list = list(word)
    results = []  # 用于存储所有交换结果，使用集合避免重复

    # 遍历单词中的每一个位置
    for i in range(len(word_list)):
        # 从当前位置之后遍历剩下的字母
        for j in range(i + 1, len(word_list)):
            # 交换字母
            word_list[i], word_list[j] = word_list[j], word_list[i]
            # 添加交换后的单词到结果集合
            results.append(''.join(word_list))
            # 交换回去以便进行下一个交换
            word_list[i], word_list[j] = word_list[j], word_list[i]

    return results

# 生成候选词
def edit(word):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    transposes = exchange_letter(word)
    return_set = set(deletes + replaces + inserts + transposes)
    return return_set

def edit_upper(word):
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    transposes = exchange_letter(word)
    return_set = set(deletes + replaces + inserts + transposes)
    return return_set


def generate_candidates(word, vocab):
    # 生成包含单个编辑操作的候选词列表
    candidates1 = edit(word)

    # 在词汇表中查找第一轮编辑的候选词
    valid_candidates1 = [w for w in candidates1 if w in vocab]
    if valid_candidates1:
        return valid_candidates1

    # 如果没有找到合适的候选词，进行第二轮编辑操作
    further_candidates = []
    for candidate in candidates1:
        further_candidates.extend(edit(candidate))

    # 在词汇表中查找第二轮编辑的候选词
    valid_candidates2 = [w for w in further_candidates if w in vocab]
    if valid_candidates2:
        return valid_candidates2

    # 如果仍然没有找到合适的候选词，尝试大写字母变体
    return generate_upper_candidates(word, vocab)


def generate_upper_candidates(word, vocab):
    # 生成包含大写字母变体的候选词列表
    candidates1_upper = edit_upper(word)

    # 在词汇表中查找第一轮大写字母编辑的候选词
    valid_candidates1_upper = [w for w in candidates1_upper if w in vocab]
    if valid_candidates1_upper:
        return valid_candidates1_upper

    # 生成进一步的大写字母变体候选词列表
    further_candidates_upper = []
    for candidate in candidates1_upper:
        further_candidates_upper.extend(edit_upper(candidate))

    # 在词汇表中查找第二轮大写字母编辑的候选词
    valid_candidates2_upper = [w for w in further_candidates_upper if w in vocab]
    if valid_candidates2_upper:
        return valid_candidates2_upper

    # 如果仍然没有找到合适的候选词，返回空列表
    print(f'无法编辑: {word}')
    return []



# 构建 N-gram 模型
def build_ngram_models(n, lambda_values):
    model = NGramModel(n, lambda_values)
    # 使用 Reuters 语料库构建 N-gram 模型
    for file_id in reuters.fileids():
        lines = reuters.open(file_id).readlines()
        model.build_ngram_model(lines)

    return model


# 拼写检查函数
def spell_check( sentence, dictionary,  model2 ):
    tokens = word_tokenize(sentence)
    corrected_tokens = []

    for i, token in enumerate(tokens):
        if token in dictionary:
            corrected_tokens.append(token)
            continue

        candidates = generate_candidates(token, dictionary)
        if not candidates:
            corrected_tokens.append(token)
            continue

        best_candidate = token  # 默认情况下，保留原词

        if i == 0:
            # 第一个词，使用 bigram 模型
            next_words = tokens[i + 1].lower() if i + 1 < len(tokens) else ''
            candidates_scores = [(w, model2.calculate_ngram_probability([w]).get(next_words, 0)) for w in candidates]
            best_candidate = max(candidates_scores, key=lambda x: x[1])[0]
        elif i < len(tokens) - 1:
            # 第三个及以后的词，使用双重 bigram 模型
            prev_word = tokens[i - 1].lower()
            next_words = tokens[i + 1].lower()
            candidates_scores1 = [(w, model2.calculate_ngram_probability([prev_word]).get(w, 0)) for w in candidates]
            candidates_scores2 = [(w, model2.calculate_ngram_probability([w]).get(next_words, 0)) for w in candidates]
            candidates_scores = [(w, score1 + score2) for (w, score1), (_, score2) in zip(candidates_scores1, candidates_scores2)]
            best_candidate = max(candidates_scores, key=lambda x: x[1])[0]
        else:
            # 最后一个词，使用 bigram 模型
            prev_word = tokens[i - 1].lower()
            candidates_scores = [(w, model2.calculate_ngram_probability([prev_word]).get(w, 0)) for w in candidates]
            best_candidate = max(candidates_scores, key=lambda x: x[1])[0]

        corrected_tokens.append(best_candidate)

    return ' '.join(corrected_tokens)


# 读取测试数据
def read_data(data_path):
    data = []
    with open(data_path, 'r') as file:
        for line in file:
            data.append(line.strip())
    return data


# 主函数
def main():
    vocab_path = 'vocab.txt'  # 假设这是包含词典的文件路径
    test_data_path = 'testdata.txt'  # 假设这是包含测试数据的文件路径
    result_path = 'result.txt'

    # 加载词典
    dictionary = load_dictionary(vocab_path)

    # 使用 Reuters 语料库构建 N-gram 模型
    model3 = build_ngram_models(3, [0.3, 0.3, 0.4])
    model2 = build_ngram_models(2, [0.4, 0.6])
    model1 = build_ngram_models(1, [1.0])

    # 加载测试数据
    test_data = read_data(test_data_path)

    # 拼写检查并输出结果
    with open(result_path, 'w') as result_file:
        for line in test_data:
            parts = line.split('\t')
            if len(parts) < 3:
                print(f"跳过格式错误的行: {line}")
                continue
            sentence_id,error_counts, sentence = parts[0],parts[1], parts[2]
            corrected_sentence = spell_check(sentence, dictionary, model2 )
            # 确保逗号后只有一个空格，并且逗号前没有空格
            corrected_sentence = corrected_sentence.replace(' ,', ',').replace(',  ', ', ')
            # 确保句子结尾的句号前没有多余的空格
            corrected_sentence = corrected_sentence.replace(' .', '.')
            # 确保所有格 's 前没有空格
            corrected_sentence = corrected_sentence.replace(" 's", "'s")
            result_file.write(f"{sentence_id}\t{corrected_sentence}\n")
        print("拼写更正完成，结果已保存到 result.txt")


if __name__ == "__main__":
    main()
    subprocess.run(["python", "eval.py"])