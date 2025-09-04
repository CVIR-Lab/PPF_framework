<div align="center">
  <h1>Dynamic-Target Potential Pursuit Field Reward for UAV Reinforcement Learning</h1>
</div>

<div align="center">
  <table>
    <tr>
      <td><img src="assets/image/process.png" width="100%"></td>
    </tr>
    <tr>
      <td align="center">The proposed PPF-based hierarchical reinforcement learning framework</td>
    </tr>
  </table>
</div>

Potential Pursuit Field (PPF), a novel reward shaping framework aimed to address the reward sparsity in reinforcement learning for dynamic target pursuit. By designing a droplet-shaped anisotropic potential field, the proposed PPF model provides dense and direction-aware reward signals while preserving policy invariance through potential-based reward shaping. Building upon PPF, we developed a hierarchical reinforcement learning algorithm, enabling target pursuit and obstacle avoidance in non-line-of-sight(NLOS) environments, simultaneously. 

## Demo
[🎬Bilibili](https://www.bilibili.com/video/BV1nDamziEjh)

[🎬 Watch demo (MP4)](./assets/video/gavideo.mp4)


## Potential Pursuit Field(PPF)

<p align="center">
  <img src="./assets/image/PPF_simulator.png" alt="Potential Pursuit Field (PPF)" width="480">
  <br>
  <em>Art work of Potential Pursuit Field (PPF)</em>
</p>


### A novel concept of the Potential Pursuit Field (PPF) is proposed to support a continuous and dense reward-shaping function, which can capture anisotropic features and obtain richer gradient information than that of traditional rewards.


## Obstacle-free pursuit
<p align="center">
  <img src="./assets/gif/NO_obs_3.gif" alt="Obstacle-free pursuit" width="49%">
  <img src="./assets/gif/NO_obs_vis_3.gif" alt="Obstacle-free pursuit (visualization)" width="49%">
</p>
<p align="center"></p>


## Obstacle pursuit

<p align="center">
  <img src="./assets/gif/OBS_1.gif" alt="Obstacle pursuit" width="49%">
  <img src="./assets/gif/OBS_1_vis.gif" alt="Obstacle pursuit (visualization)" width="49%">
</p>
<p align="center"></p>





## This is the official for manuscript entitled Dynamic-Target Pursuit Potential Field Reward for UAV Reinforcement Learning submitted to IEEE Transactions on Control Systems Technology

The extire code and corresponding simulation environment will be released later.
