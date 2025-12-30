import csv
import os
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import serial
import tqdm

from ballisticgel import CW521
from chipshouter import ChipSHOUTER

MAX_SIZE = 4 * 1024 * 1024


class Control:
    def __init__(self, x_min, x_max, y_min, y_max):
        self.step = 1
        self.path = None
        self.plot_data = []
        self._speed = 500

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        self.x = None
        self.y = None

    # ChipSHOUTER settings
    def cs_setup(self):
        cs = ChipSHOUTER("COM4")
        cs.armed = 0
        cs.pulse.repeat = 1
        cs.voltage = 300
        cs.pulse.width = 160

        return cs

    # generates surface scan GCODE
    def gen_scan(self, x_start, x_stop, y_start, y_stop, step):
        if not (
            self.x_min <= x_start <= self.x_max and self.y_min <= y_start <= self.y_max
        ):
            raise ValueError("Out of boundry")
        if not (
            self.x_min <= x_stop <= self.x_max and self.y_min <= y_stop <= self.y_max
        ):
            raise ValueError("Out of boundry")

        x = np.arange(x_start, x_stop, step)
        y = np.arange(y_start, y_stop, step)

        self.path = []
        self.plot_data = []
        for xi in x:
            for yi in y:
                self.path.append(f"G0 X{round(xi,1)} Y{round(yi,1)} F{self._speed}\n")
                self.plot_data.append([xi, yi])
            y = y[::-1]

    # executes genreated GCODE
    def run_scan(self):

        x_data = []
        y_data = []
        e_data = []
        cw521 = CW521()
        cw521.con()

        cs = self.cs_setup()

        print(cs)

        cs.armed = 1

        with serial.Serial("COM3", 115200, timeout=1) as pos_device:
            # set absolute positioning
            start = time.time()
            pos_device.reset_input_buffer()
            pos_device.write(b"G90\n")
            pos_device.readline()

            with open("scan.csv", "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["x", "y", "corr_rate"])

                pattern = bytearray(np.random.bytes(MAX_SIZE))

                # iterate throught generated path
                for i in self.path:
                    print("total:", time.time() - start)
                    pos_device.write(i.encode())
                    pos_device.readlines()
                    pos_device.write(b"M400\n")
                    while True:
                        line = pos_device.readline().decode().strip()
                        # print(line)
                        if line == "ok":
                            break
                    try:
                        pas = time.time()
                        print(i, end="")
                        cw521.write_pattern(pattern)
                        time.sleep(0.1)
                        cs.pulse = 1
                        time.sleep(0.5)
                        data = cw521.read_pattern()
                        diff = np.bitwise_xor(pattern, data)

                        location = i.split(" ")
                        # print(location)

                        error_num = np.count_nonzero(diff)

                        writer.writerow(
                            [
                                float(location[1][1:]),
                                float(location[2][1:]),
                                error_num / MAX_SIZE,
                            ]
                        )

                        # if np.count_nonzero(diff) != 0:

                        x_data.append(float(location[1][1:]))
                        y_data.append(float(location[2][1:]))
                        e_data.append(error_num / MAX_SIZE)
                        plt.clf()
                        plt.scatter(x_data, y_data, c=e_data)
                        plt.colorbar()
                        plt.gca().set_aspect("equal", "datalim")
                        plt.savefig("run.png")
                        print("*" * 10, error_num)
                        print("pass:", time.time() - pas)
                    except:
                        cs.armed = 0
                        cw521.close()
                        raise
            cs.armed = 0
            cs.disconnect()
            cw521.close()
            print("Run time:", time.time() - start)

    # saves an image of path
    def plot_scan(self):
        plot = np.array(self.plot_data)
        plt.plot(plot[:, 0], plot[:, 1])
        plt.gca().set_aspect("equal", adjustable="box")
        plt.title(f"{len(self.plot_data)} | {len(self.plot_data)*2}s")
        plt.savefig("map.png")

    # moves positional device to specified location
    def move_2D(self, x, y):
        if self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max:
            with serial.Serial("COM3", 115200, timeout=2) as printer:
                printer.reset_input_buffer()
                print("Move started.")

                printer.write(b"G90\n")
                printer.readline()

                gcode = f"G0 X{round(x,1)} Y{round(y,1)} F{self._speed}\n"
                printer.write(gcode.encode())
                printer.readline()

                printer.write(b"M400\n")
                while printer.readline().decode().strip() != "ok":
                    print("waiting for move to finish")

                print("Move finished.")

                self.x = x
                self.y = y
        else:
            print("! Move out of the boundary. !")

    # writes random or defined data and attack them.
    def random_read(self, repeat=1, data: bytearray = None):
        cw521 = CW521()
        cw521.con()

        cs = self.cs_setup()
        cs.armed = 1

        for _ in range(repeat):
            if data is None:
                pattern = bytearray(np.random.bytes(MAX_SIZE))
            else:
                pattern = data
            try:
                cw521.write_pattern(pattern)
                time.sleep(0.1)
                cs.pulse = 1
                time.sleep(0.5)
                cor_pattern = cw521.read_pattern(start_addr=0, len_to_read=len(pattern))
                byte_flips = np.bitwise_xor(
                    np.frombuffer(pattern, dtype=np.uint8),
                    np.frombuffer(cor_pattern, dtype=np.uint8),
                )
                print(np.count_nonzero(byte_flips), end=", ")
                print(np.count_nonzero(byte_flips) / len(pattern) * 100)
                print(cor_pattern[:32])
            except:
                break

        cs.armed = 0
        cs.disconnect()
        cw521.close()

        return byte_flips, np.count_nonzero(byte_flips)

    # used for corruption of weights
    def corrupt_dump(self, save_folder_name=None):
        if save_folder_name == None:
            return

        cw521 = CW521()
        cw521.con()

        cs = self.cs_setup()

        (Path.cwd() / save_folder_name).mkdir(exist_ok=True)

        (Path.cwd() / save_folder_name / "run_info.txt").write_text(
            "x:" + str(self.x) + "y:" + str(self.y) + "\n" + str(cs)
        )

        cs.armed = 1

        path = Path.cwd() / "model_chunks"
        for file_to_corr in tqdm.tqdm(list(path.rglob("*.bin"))):
            save_path = (
                Path.cwd()
                / save_folder_name
                / file_to_corr.parent.parent.stem
                / file_to_corr.parent.stem
            )
            save_path.mkdir(exist_ok=True, parents=True)
            save_path = save_path / file_to_corr.name

            data = file_to_corr.read_bytes()
            try:
                print(file_to_corr)
                cw521.write_pattern(data)
                time.sleep(0.1)
                cs.pulse = 1
                time.sleep(0.5)
                corrupted = cw521.read_pattern()[: len(data)]

            except:
                cs.armed = 0
                cw521.close()
                raise

            save_path.write_bytes(corrupted)
            if data != corrupted:
                print("*" * 9)

            print(
                np.count_nonzero(
                    np.bitwise_xor(
                        np.frombuffer(data, dtype=np.int8),
                        np.frombuffer(corrupted, dtype=np.int8),
                    )
                )
                / len(data)
                * 100,
            )

        cs.armed = 0
        cs.disconnect()
        cw521.close()

    def set_step(self, step):
        self.step = step

    def save(self):
        with open("test.gcode", "w") as file:
            file.writelines(self.path)
