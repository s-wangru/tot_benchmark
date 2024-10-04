filename = 'traces.pkl'

import pickle as pkl 

traces = pkl.load(open(filename, "rb"))

print(len(traces))


for i, trace in enumerate(traces):
    print('id:', i)
    # print('messages:', trace.messages)
    print('len(messages):', len(trace.messages))
    # print('response:', trace.response)


print(traces[-1].messages)
# print(trace.messages[-1])