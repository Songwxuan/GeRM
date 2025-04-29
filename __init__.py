import os
from datetime import datetime
import pytz

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

DAY_TIME = datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d-%H-%M-%S")