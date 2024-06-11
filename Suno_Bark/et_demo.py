# import re
# text = ("\"So it’s a Full Spectrum Collagen Supplement! Covers allthe basesfor Hair, Skin, Nails, Joints.\""
#         "assist So far, we've discussed several key aspects of this highly effective supplement, "
#         "but here are some additional reasons that set it apart.- So, yes, it truly is. "
#         "So-it-is-a-full-spectrum-collar-ogen-supplement-Covers-all-the-bases-for-hair-skin-nails-joints. "
#         "It makes sense, given its unique composition, which includes five different types-of-collogen."
#         "` Guys, let me tell you. type one collagen makes up.for us because it supports. and ` sType  collagen?")
# arr_list = re.split('\.|\?|!', text)
# print(len(arr_list), arr_list)
max_num = 3
arr = ['ab', 'pq', 'rs', 'pq', 'rs', 'pq', 'rs']
arr[max_num] = ''
print(arr)
print('.'.join(arr[:max_num+1]))
