### Task Instruction
Assemble the cardboard box by erecting the flat sheet and folding the side flaps.

(解析：原指令只说了“折叠形成盒子”，不够具体。新指令明确了两个核心动作：1. 将平铺纸板竖立起来 (erecting)；2. 折叠侧翼 (folding side flaps)。这更准确地概括了全过程。)

### Subtask Descriptions
1. Lift the flat cardboard sheet with both arms and maintain a vertical orientation.

  (对应：拿起纸片使其保持竖直状态。视频 00:00 - 00:10，机器人双臂配合将纸板从平面拉起并竖立，这是成型的关键第一步。)

2. Fold the right side flap of the upright box using the right arm.

  (对应：右手折盒子的右边“耳朵”。视频 00:14 - 00:25，右臂主要操作右侧的折翼（flap），将其向内折叠。显式标注右臂操作，便于VLA模型进行双手机械臂的信用分配与动作解耦。)

3. Fold the left side flap of the upright box using the left arm.

  (对应：左手折盒子的左边“耳朵”。视频 00:26 - 00:45，左臂主要操作左侧的折翼，将其向内折叠，右臂此时主要起辅助支撑作用。明确左右手分工可避免策略解码时的指令冲突。)

4. Place the assembled box on the table and release the grippers.

  (对应：放下盒子。视频结尾，机器人完成折叠后，将成型的盒子稳固放置在桌面上并松开夹爪。显式包含释放动作符合工业流程闭环与安全规范。)