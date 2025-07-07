# NexusNet: Lightweight Graph Modeling for Motor Imagery-based Brain-computer Interfaces

This is a PyTorch implementation of NexusNet for MI decoding.

**All code for our NexusNet has been released. If you are interested in our work, please consider citing it.**

### Abstract
- We propose a lightweight GNN, NexusNet, designed to
capture complex relationships beyond pairwise connections.

- We conduct thorough experiments on two public datasets
to validate NexusNet. Specifically, it achieves an average
accuracy of 78.78% (hold-out) on the BCIC-IV-2a dataset and 87.21% (hold-out)
on the BCIC-IV-2b dataset.

- We visualize the primary Nexuses to quantitatively analyze
the relationships reconstructed by NexusNet. This visualization
enables a detailed examination of how different
Nexuses contribute to the decoding process.

![Framework](./framework.jpg)

### Requirements

Please refer to [requirements.txt](./requirement.txt)

### Model Zoos

Pretrained checkpoints are available in
- [bciciv2a_checkpoint](./bciciv2a_checkpoint/)
- [bciciv2b_checkpoint](./bciciv2b_checkpoint/)

### License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

### Citation

```
@article{wang2025nexusnet,
  title={NexusNet: Lightweight Graph Modeling for Motor Imagery-Based Brain-Computer Interfaces},
  author={Wang, Zikai and Si, Yuan and Wang, Zhenyu and Zhou, Ting and Xu, Tianheng and Hu, Honglin},
  journal={IEEE Internet of Things Journal},
  year={2025},
  publisher={IEEE}
}
```