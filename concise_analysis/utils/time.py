from datetime import datetime, timedelta


progress = 0
total_work = 0

start_time = None
end_time = None


def start_timer():
    global start_time, endtime
    start_time = datetime.now()
    endtime = start_time


def set_work(total_work_count):
    global progress, total_work
    progress = 0
    total_work = total_work_count


def print_progress(change=0, end=""):
    global progress, total_work, start_time, endtime
    progress += change
    end_time = datetime.now()
    tdelta = end_time - start_time
    tdelta = tdelta - timedelta(microseconds=tdelta.microseconds)
    print(
        f"\rprogress {round(100 * progress / total_work, 2)}% – "
        + "Duration: {}        ".format(tdelta),
        end=end,
    )
