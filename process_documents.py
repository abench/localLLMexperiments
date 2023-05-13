# Load environment variables

from dotenv import load_dotenv
import os

load_dotenv()
t = os.environ["test"]

print(f't:{t}')