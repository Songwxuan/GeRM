import random


def map_index_to_after_index(index, episode_length):
    after_index = (index + episode_length - 6) % (episode_length - 6)
    return after_index

episode_length = 12  # 替换为您的实际episode_length值
sampled_indices = range(episode_length)  # 从0到episode_length-1随机采样索引

mapped_indices = []
for index in sampled_indices:
    after_index = map_index_to_after_index(index, episode_length)
    mapped_indices.append(after_index)

print(mapped_indices)

# 枚举所有可能的after_index取值
after_index_counts = [0] * (episode_length - 5)
for index in range(episode_length):
    after_index = map_index_to_after_index(index, episode_length)
    after_index_counts[after_index] += 1

# 计算概率
probabilities = [count / episode_length for count in after_index_counts]

# 打印结果
for after_index, probability in enumerate(probabilities):
    print(f"after_index={after_index}: probability={probability}")