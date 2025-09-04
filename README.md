<div align="center">
  <img src="assets/image/zhaiyaotu.png" alt="Header Image" width="100%"/>
  <br/>
  <h1>PPF-framework: Dynamic-Target Potential Pursuit Field Reward for UAV Reinforcement Learning</h1>
</div>

Potential Pursuit Field (PPF),a novel reward shaping framework aimed to address the reward sparsity in reinforcement learning for dynamic target pursuit. By designing a droplet-shaped anisotropic potential field, the proposed PPF model provides dense and direction-aware reward signals while preserving policy invariance through potential-based reward shaping. Building upon PPF, we developed a hierarchical reinforcement learning algorithm, enabling target pursuit and obstacle avoidance in non-line-of-sight(NLOS) environments, simultaneously. 

## Demo
[🎬 Watch demo (MP4)](./assets/video/gavideo.mp4)

## Obstacle-free pursuit
<div align="">
  <table>
    <tr>
      <td><img src="assets/image/PPF_simulator.png" width="40%"></td>
    </tr>
    <tr>
      <td align="center">Potential Pursuit Field(PPF)</td>
    </tr>
  </table>
</div>

 A novel concept of the Potential Pursuit Field (PPF) is proposed to support a continuous and dense reward-shaping function, which can capture anisotropic features and obtain richer gradient information than that of traditional rewards.

![singlecontrol](assets/gif/No_obs_3.gif)
### visulization
![singlecontrol](assets/gif/No_obs_vis_3.gif)



## Obstacle pursuit
<div align="center">
  <table>
    <tr>
      <td><img src="assets/image/process.png" width="100%"></td>
    </tr>
    <tr>
      <td align="center">PPF-based framework</td>
    </tr>
  </table>
</div>

A hierarchical reinforcement learning algorithm is proposed based on PPF, which can pursuit the NLOS target under obstacle environment.
 
![singlecontrol](assets/gif/Obs_1.gif)
### visulization
![singlecontrol](assets/gif/Obs_1_vis.gif)



## This is the official for manuscript entitled Dynamic-Target Pursuit Potential Field Reward for UAV Reinforcement Learning submitted to IEEE Transactions on Control Systems Technology

The extire code and corresponding simulation environment will be released later.
