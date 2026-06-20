from wl100_demo.nlu_schema import STAGE_NAMES, StageAction
from wl100_demo.skill_mapper import build_action_reply


class TaskExecutor:
    """统一执行出口：回复先播，播完才下发动作。"""

    def __init__(self, publish_reply, wait_for_reply, publish_stage,
                 publish_cancel, publish_mission_end, logger):
        self.publish_reply = publish_reply
        self.wait_for_reply = wait_for_reply
        self.publish_stage = publish_stage
        self.publish_cancel = publish_cancel
        self.publish_mission_end = publish_mission_end
        self.logger = logger

    def dispatch(self, action: StageAction, reply: str = "",
                 user_request: str = "") -> bool:
        stage = action.stage
        if stage == "cancel":
            self.publish_cancel()
            self.publish_reply(reply or "已取消当前任务")
            return True

        spoken = reply or build_action_reply([action])
        if not self.wait_for_reply(spoken):
            self.logger.warn(f"动作 {stage} 前播报被取消，跳过分发")
            return False

        if stage == "generate_report":
            self.publish_mission_end()
            return True

        if stage in STAGE_NAMES:
            self.publish_stage(
                stage,
                action.target_object,
                user_request,
                waypoint=action.waypoint,
                target_description=action.target_description,
                target_type=action.target_type,
            )
            return True

        self.logger.warn(f"未知动作 {stage!r}，跳过")
        return False
