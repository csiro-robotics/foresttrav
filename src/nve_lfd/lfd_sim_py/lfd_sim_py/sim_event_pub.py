
from pickletools import uint8
import rclpy
from rclpy.node import Node

from lfd_event_msg.msg import LfdEvent
from std_msgs.msg import UInt8


class LfdRcEventPublisherNode(Node):

    def __init__(self) -> None:
        super().__init__("lfd_rc_event_node")

        self.trigger_state_ = -1 # Trigger is responsble if we want to record
        self.switch_state_ = -1  # Logger determins what we record
        self.time_stamp_ = -1    # Time stamp allows us to understand when we recorded something

        # Event publisher
        self.event_pub_ = self.create_publisher(LfdEvent, 'event', 5)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.event_pub_cb)
        
        # Subscribers
        trigger_topic = "/squash/logging_trigger"
        logging_topic = "/squash/logging_state"
        self.sub_trigger = self.create_subscription( UInt8, trigger_topic,self.trigger_cb, 5)
        self.sub_logger = self.create_subscription( UInt8, logging_topic,self.logger_cb, 5)
    

    def event_pub_cb(self):
            """ Callback to populate the event msg"""
            msg = LfdEvent()
            msg.header.frame_id = "map"
            msg.header.stamp = self.get_clock().now().to_msg()
            
            if self.trigger_state_ == 2 and self.switch_state_ == 0:
                msg.event_cost = 1
                msg.event_type = 2

            elif self.trigger_state_ == 2 and self.switch_state_ == 2:
                msg.event_cost = -1
                msg.event_type = 3

            self.event_pub_.publish(msg)
   
    def trigger_cb(self, msg):
            self.trigger_state_ = msg.data

    def logger_cb(self,msg):
            self.switch_state_ = msg.data



def main(args=None):
    rclpy.init(args=args)
    node = LfdRcEventPublisherNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
