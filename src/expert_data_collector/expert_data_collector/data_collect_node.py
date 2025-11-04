import rclpy
from rclpy.node import Node


class DataCollectNode(Node):
    def __init__(self):
        super().__init__('data_collect_node')


def main(args=None):

    rclpy.init(args=args)
    node = DataCollectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()