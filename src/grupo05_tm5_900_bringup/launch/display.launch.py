from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable, TimerAction
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    model = PathJoinSubstitution([
        FindPackageShare("tm_description"),
        "xacro",
            "tm5-900.urdf.xacro"
    ])
    robot_description = ParameterValue(
        Command([
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            model,
        ]),
        value_type=str,
    )
    rviz_config = PathJoinSubstitution([
        FindPackageShare("grupo05_tm5_900_bringup"),
        "rviz",
        "display.rviz",
    ])

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description}],
    )
    gui = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
        name="joint_state_publisher_gui",
        output="screen",
    )
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
    )

    return LaunchDescription([
        SetEnvironmentVariable("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp"),
        rsp,
        rviz,
        TimerAction(period=1.0, actions=[gui]),
    ])
