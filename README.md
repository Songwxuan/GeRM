# GeRM: A Generalist Robotic Model with Mixture-of-Experts for Quadruped Robot

Welcome to the official repository of **GeRM: A Generalist Robotic Model with Mixture-of-Experts for Quadruped Robot**. 🤖🐾

This repository contains the complete code for the data processing, training, and testing pipelines of GeRM. Some of the model architecture code is modified from [Robotics Transformer by Google Research](https://github.com/google-research/robotics_transformer). 💻✨

We hope this code helps the community in advancing the field of robotics! 🚀

## Features

- 🗂️ **Data Preprocessing:**  
  The data preprocessing code is located in `data_process.sh`. 

- 🏋️‍♂️💻 **Training Script:**  
  The training script is located in `train_ddp.sh`, and it supports multi-GPU training. 

- 🏗️ **Model Architecture:**  
  The model architecture code is located in `pytorch_robotics_transformer/transformer_network.py`. 

- 🧪 **Core Testing Code:**  
  The core testing code is located in `pytorch_robotics_transformer/transformer_inference.py`. 

- 🎮 **Agent Training Script:**  
  The agent training code is located in `agent_ddp.py`.

## Future Directions

💡 One promising direction for future work is to extend our code to apply **Visual Language Attention (VLA)** training to robotic arms. 🤖

## Environment Configuration and Dataset

🛠️📦 The environment setup in Isaac Gym is relatively complex and requires extensive configuration. We plan to open source the environment setup and dataset code in the future to make it more accessible. 

## Issues and Contributions

💬 If you have any questions or issues, feel free to leave a message in the **Issues** section. We'd love to hear your thoughts and feedback! 

## Acknowledgments

🙏💡 We would like to thank Google Research for their incredible work on the [Robotics Transformer](https://github.com/google-research/robotics_transformer), which provided the foundational model architecture for GeRM.

---

Stay tuned for future updates, and happy coding! 🎉

## Citation

If you find this work useful, please consider citing the following paper:

```bibtex
@inproceedings{song2024germ,
  title={Germ: A generalist robotic model with mixture-of-experts for quadruped robot},
  author={Song, Wenxuan and Zhao, Han and Ding, Pengxiang and Cui, Can and Lyu, Shangke and Fan, Yaning and Wang, Donglin},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={11879--11886},
  year={2024},
  organization={IEEE}
}