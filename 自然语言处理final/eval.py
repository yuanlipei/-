# import nltk
# anspath='./ans.txt'
# resultpath='./result.txt'
# ansfile=open(anspath,'r')
# resultfile=open(resultpath,'r')
# count=0
# for i in range(1000):
#     ansline=ansfile.readline().split('\t')[1]
#     ansset=set(nltk.word_tokenize(ansline))
#     resultline=resultfile.readline().split('\t')[1]
#     resultset=set(nltk.word_tokenize(resultline))
#     if ansset==resultset:
#         count+=1 ##sb
# print("Accuracy is : %.2f%%" % (count*1.00/10))

import nltk

anspath = './ans.txt'
resultpath = './result.txt'
ansfile = open(anspath, 'r')
resultfile = open(resultpath, 'r')
count = 0
wrong_words = []

for i in range(1000):
    ansline = ansfile.readline().strip().split('\t')[1]
    ansset = set(nltk.word_tokenize(ansline))
    resultline = resultfile.readline().strip().split('\t')[1]
    resultset = set(nltk.word_tokenize(resultline))

    if ansset != resultset:
        wrong_words.append((i + 1, list(ansset - resultset), list(resultset - ansset)))
    else:
        count += 1

accuracy = count / 1000
print(f"Accuracy is: {accuracy:.2%}")

if wrong_words:
    print("\nIncorrect comparisons:")
    for line_num, ans_diff, result_diff in wrong_words:
        print(f"Line {line_num}:")
        print(f"  Answer extra words: {ans_diff}")
        print(f"  Result extra words: {result_diff}")
        print()
else:
    print("\nAll comparisons are correct.")
