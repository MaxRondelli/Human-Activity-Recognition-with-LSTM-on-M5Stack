from enum import Enum

class Activity(Enum):
    WALKING = 1
    WALKING_UPSTAIRS = 2
    WALKING_DOWNSTAIRS = 3
    SITTING = 4
    STANDING = 5
    LAYING = 6

class_mapping = {activity.name: activity.value for activity in Activity}