## Dynamic Transmission Policy for Enhancing LoRa Networks Performance: A Deep Reinforcement Learning Approach
### Abstract 
Long Range (LoRa) communications, operating through the LoRaWAN protocol, have received increasing attention from the low-power and wide-area
networks community. Efficient energy consumption and reliable communication performance are critical concerns in LoRa-based applications. However,
current scientific literature tends to mainly focus on minimizing energy consumption while disregarding channel changes affecting communication performance. On the other hand, other works attain the appropriate communication performance without adequately considering energy expenditure. To
fill this gap, we propose a novel solution to strike a balance between energy consumption and communication performance metrics. In particular,
we characterize the problem as a Markov Decision Process and solve it using Deep Reinforcement Learning algorithms to dynamically select the transmission parameters that jointly satisfy energy and performance requirements
over time. We evaluate the performance of the proposed algorithm in three different scenarios by comparing it with the traditional Adaptive Data Rate
(ADR) mechanism of LoRaWAN. 

### Available models 
Available trained models are ready to be used in the log folder.

### Requirements 
- Python 3.10
- gym 0.21.0
- matplotlib 3.5.1
- numpy 1.23.4
- pandas 1.4.3
- scikit-learn 1.2.1
- seaborn 0.12.2
- stable-baselines3 1.7.0

### Paper published
Acosta-Garcia, Laura, et al. "Dynamic transmission policy for enhancing LoRa network performance: A deep reinforcement learning approach." Internet of Things 24 (2023): 100974.
[Link to the published paper](https://www.sciencedirect.com/science/article/pii/S2542660523002974)
