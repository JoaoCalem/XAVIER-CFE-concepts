# XAVIER-CFE-concepts

## Setup
```
pip install git+https://github.com/JoaoCalem/XAVIER-CFE-concepts
```

## Example Code

```
import xavierconcepts
df = xavierconcepts.getConcepts('path-where-files-will-be-downloaded-to')
```

### Other Examples

```
df = xavierconcepts.getConcepts('.', target=1, v_clip_name='c_texture')
```

```
model = xavierconcepts.getClassifier('.')
```

```
labels, v_clip = xavierconcepts.getVClip('.', name='c_color')
```

## Reference to Paper
Kim, S., Oh, J., Lee, S., Yu, S., Do, J., & Taghavi, T. (2023). Grounding Counterfactual Explanation of Image Classifiers to Textual Concept Space. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 10942–10950). IEEE.