import curses
import rclpy
import threading
import time
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String
from typing import get_args, Literal, TypeAlias

JOINT_NUMS: int = 10  # exclude back, sacrum
TeleopKey: TypeAlias = Literal['w', 'a', 's', 'd', 'e', 'r']
ALLOWED_TELEOPKEYS = set(get_args(TeleopKey))


class BipedKeyboardNode(Node):
    def __init__(self, stdscr):
        super().__init__('biped_keyboard_node')

        self._joint_target_publisher_ = self.create_publisher(
            Float64MultiArray,
            '/biped/joint_target',
            10
        )
        self._teleop_key_publisher_ = self.create_publisher(
            String,
            '/biped/teleop_key',
            10
        )

        self._key_in_count: int = 0
        self._stdscr = stdscr
        self._stdscr.keypad(False)

    def process_key_in(self):
        #   lower-case  -> IK/teleop mode: interpret as motion intents
        #   UPPER-CASE  -> Direct-Joint mode: publish joint targets directly (no IK)
        #   ('q' quits)
        while rclpy.ok():
            input_key: int = self._stdscr.getch()
            if input_key == curses.ERR:
                self._print_basic_info(ord(' '))
                time.sleep(0.1)
                continue
            self._key_in_count += 1
            self._print_basic_info(input_key)

            if not (0 <= input_key < 256):
                continue
            input_key_chr: str = chr(input_key)
            if input_key_chr.lower() == 'q':  # Exit
                break
            elif input_key_chr in ALLOWED_TELEOPKEYS:
                key: TeleopKey = input_key_chr
                self._pub_teleop_key(key)
            elif input_key_chr == 'W':
                joint_pos: list[float] = [0.0]*JOINT_NUMS
                self._pub_joint_pos(joint_pos)

    def _print_basic_info(self, key):
        self._stdscr.clear()
        self._stdscr.move(0, 0)
        self._stdscr.addstr(
            f"{self._key_in_count:5d} Key '{chr(key)}' pressed!")

    def _pub_teleop_key(self, key) -> None:
        msg = String()
        msg.data = key
        self._teleop_key_publisher_.publish(msg)

    def _pub_joint_pos(self, joint_pos) -> None:
        msg = Float64MultiArray()
        msg.data = [float(i) for i in joint_pos]
        self._joint_target_publisher_.publish(msg)


def main(args=None):
    stdscr = curses.initscr()
    curses.noecho()
    curses.raw()

    rclpy.init(args=args)
    biped_keyboard_node = BipedKeyboardNode(stdscr)
    spin_thread = threading.Thread(
        target=rclpy.spin,
        args=(biped_keyboard_node,),
        daemon=True)
    spin_thread.start()

    try:
        biped_keyboard_node.process_key_in()
    finally:
        biped_keyboard_node.get_logger().info(f'Quit')
        curses.endwin()
        rclpy.shutdown()
        spin_thread.join()


if __name__ == '__main__':
    main()
