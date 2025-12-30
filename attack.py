from hardtools.control import Control

# define the boundaries of movement
control = Control(76, 86, 142, 163)

# surface scan of the chip
if 1:
    control.gen_scan(76, 86, 142, 163, 0.5)  # full chip scan

    control.plot_scan()
    control.run_scan()


# test locations on random data and calculate the corruption rate
if 0:
    x = [79.5, 80, 80, 80, 80]
    y = [160.5, 160, 159, 158, 157]

    for pos in zip(x, y):
        print("*" * 10, pos[0], pos[1])
        control.move_2D(pos[0], pos[1])
        control.random_read(5)

# corrupt contents of folder model_chunks
if 0:
    control.move_2D(79.5, 160.5)
    control.corrupt_dump("model_chunks_corr_2")
