from envs._base_task import Base_Task
from envs.pick_diverse_bottles import pick_diverse_bottles
from envs.utils import *
import sapien

class gpt_pick_diverse_bottles(pick_diverse_bottles):
    def play_once(self):
        # Initial scene observation
        self.save_camera_images(task_name="pick_diverse_bottles", step_name="step1_initial_scene", generate_num_id="generate_num_0")

        # Define arm tags
        left_arm_tag = ArmTag("left")
        right_arm_tag = ArmTag("right")

        # Grasp both bottles simultaneously
        self.move(
            self.grasp_actor(actor=self.bottle1, arm_tag=left_arm_tag),
            self.grasp_actor(actor=self.bottle2, arm_tag=right_arm_tag)
        )
        self.save_camera_images(task_name="pick_diverse_bottles", step_name="step2_grasp_bottles", generate_num_id="generate_num_0")

        # Lift both bottles up to avoid collision
        self.move(
            self.move_by_displacement(arm_tag=left_arm_tag, z=0.07),
            self.move_by_displacement(arm_tag=right_arm_tag, z=0.07)
        )
        self.save_camera_images(task_name="pick_diverse_bottles", step_name="step3_lift_bottles", generate_num_id="generate_num_0")

        # Move both bottles to their respective target locations
        self.move(
            self.place_actor(
                actor=self.bottle1,
                arm_tag=left_arm_tag,
                target_pose=self.left_target_pose,
                functional_point_id=0,
                pre_dis=0.1,
                dis=0.02,
                is_open=False,  # Keep the gripper closed
                pre_dis_axis='fp'
            ),
            self.place_actor(
                actor=self.bottle2,
                arm_tag=right_arm_tag,
                target_pose=self.right_target_pose,
                functional_point_id=0,
                pre_dis=0.1,
                dis=0.02,
                is_open=False,  # Keep the gripper closed
                pre_dis_axis='fp'
            )
        )
        self.save_camera_images(task_name="pick_diverse_bottles", step_name="step4_move_to_target", generate_num_id="generate_num_0")

        # Optionally, lift both arms slightly after placing to ensure no collision
        self.move(
            self.move_by_displacement(arm_tag=left_arm_tag, z=0.07),
            self.move_by_displacement(arm_tag=right_arm_tag, z=0.07)
        )

        # Final scene observation
        self.save_camera_images(task_name="pick_diverse_bottles", step_name="step5_final_scene", generate_num_id="generate_num_0")

'''
Observation Point Analysis:
1. Initial scene state
2. Grasp both bottles simultaneously
3. Lift both bottles up to avoid collision
4. Move both bottles to their respective target locations
5. Final scene state
'''
