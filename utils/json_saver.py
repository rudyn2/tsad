import json
import os


class JsonSaver:
    def __init__(self, path: str):
        self.path = path

    def write(self, d: dict):
        if not os.path.exists(self.path):
            with open(self.path, "w+") as file:
                json.dump(d, file)
                return

        with open(self.path, "r+") as file:
            data = json.load(file)
            data.update(d)
            file.seek(0)
            json.dump(data, file)

    def save_one_ego_run(self, info_data: list, run_id: str):

        tries = 3
        while tries > 0:
            try:
                # ego info
                ego_run_info = {run_id: {}}
                for item in info_data:
                    ego_run_info[run_id][item['timestamp']] = item['metadata']
                self.write(ego_run_info)
                break
            except json.JSONDecodeError:
                pass
            tries -= 1
