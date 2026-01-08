import asyncio
from visualizer import Visualizer

if __name__ == "__main__":
    app = Visualizer()
    asyncio.run(app.run())