from wordsegment import load, segment
load()
print(segment('andriod123app'))
print(segment(''.join(list(reversed('andriodapp')))))
import re
ge=[r for r in re.split('[^(a-zA-Z)]','andriod123app') if len(r)>0]
ge=[r for r in re.split('[^(a-zA-Z0-9)]','andriod表示匹配123app') if len(r)>0]

print(ge)
ag='andriod123app表示'
for t in ge:
    ag=ag.replace(t,' ')
print(ag)

