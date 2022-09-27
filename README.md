# About This Repository

This repository is meant to host utilities and tools for TF38xx Machine Learning development. The current tools within this repository are already included with the [TF38xx installer](https://www.beckhoff.com/en-us/products/automation/twincat/tfxxxx-twincat-3-functions/tf3xxx-tc3-measurement/tf3800.html) from Beckhoff Automation. The GitHub hosting of these files is meant to help streamline the development process with data science notebooks (Google Colab, Deepnote, AWS Sagemaker) and Docker containers.



# Example Use


#### Google Colab:

```python
def import_TF38xx_Utilities():
  # import required modules
  import os, sys
  # repo info
  user = "Beckhoff-USA-Community"
  repo = "TF38xx-Machine-Learning-Utilities"
  src_dir = "src"

  # if repo is already cloned, remove and clone again
  if os.path.isdir(repo):
      !rm -rf {repo}

  # clone repo
  !git clone https://github.com/{user}/{repo}.git --quiet

  # add modules to path
  path = f"{repo}/{src_dir}"
  if not path in sys.path:
      sys.path.insert(1, path)
```

```python
# Import the Beckhoff modules
import_TF38xx_Utilities()

from ScikitLearnSvm2Xml import svm2xml
```







# How to get support

Should you have any questions regarding the provided sample code, please contact your local Beckhoff support team. Contact information can be found on the official Beckhoff website at https://www.beckhoff.com/en-us/support/.

