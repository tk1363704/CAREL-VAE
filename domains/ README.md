THUCTC_multiple: the dataset in which an utterance might have multiple emotions.
For instance:

434 11

 (6,5), (11,11)

1,null,null,大年初一 失踪 的 4 岁 女孩

2,null,null,40 多天 后 又 被 送 了 回来

3,null,null,前天 早上 6 时 40 分许

4,null,null,在 新安县 尤彰村 310 国道 附近

5,null,happiness,女孩 郭佳诺 的 归来

6,happiness,惊喜,让 亲戚 惊喜不已

7,null,null,而 这个 地方 正是 小佳诺 之前 失踪 的 地方

8,null,null,目前

9,null,null,小佳诺 精神状态 良好

10,null,null,全身 未 发现 有 被 伤害 的 痕迹

11,fear,fear,但 谈及 过去 仍 不 时会 有 惊恐 表情

---
THUCTC: the dataset in which an utterance only have one emotion.

For instance:

1434 11

 (6,5), (11,11)

1,null,null,大年初一 失踪 的 4 岁 女孩

2,null,null,40 多天 后 又 被 送 了 回来

3,null,null,前天 早上 6 时 40 分许

4,null,null,在 新安县 尤彰村 310 国道 附近

5,null,null,女孩 郭佳诺 的 归来

6,happiness,惊喜,让 亲戚 惊喜不已

7,null,null,而 这个 地方 正是 小佳诺 之前 失踪 的 地方

8,null,null,目前

9,null,null,小佳诺 精神状态 良好

10,null,null,全身 未 发现 有 被 伤害 的 痕迹

11,fear,惊恐,但 谈及 过去 仍 不 时会 有 惊恐 表情

---
Within THUCTC_multiple, the domain society_num convert all the emotion tokens into ID, for instance NULL into 6 and happiness into 0.

---
