class ScheduleManager:
    def __init__(self, queue, interval=10):
        super().__init__()

        self.next_id = 0
        self.current = None  # TODO implement a system that chooses one process and shows the states from it
        self.tracking = {}
        self.interval = interval
        self.pid_by_idx = []
        self._first = True
        self.queue = queue

    def register_new_process(self, process, client):
        pid = process.pid
        idx = len(self.pid_by_idx)
        self.tracking[pid] = Tracked(process, client, idx)
        self.pid_by_idx.append(pid)
        if self._first:
            self._first = False
            client.flag = True  # set first client as true
    
    def announce_flag(self, pid):
        tracked = self.tracking[pid]
        self.current = tracked
        if tracked.time_with_flag >= self.interval:
            tracked.client.flag = False
            idx = (tracked.idx + 1) % len(self.pid_by_idx)  # get index of next process
            pid = self.pid_by_idx[idx]
            next_tracked = self.tracking[pid]
            next_tracked.client.flag = True
            self.current = next_tracked
            tracked.time_with_flag = 0
        else:
            tracked.time_with_flag += 1
    
    def check_all_flags(self):
        if self.current is not None:
            for _, tracked in self.tracking.items():
                if tracked.client.flag:
                    # if one of the tracked clients' flag is True it means that client is alive and running
                    # else assign true to the next client and continue
                    return True
            idx = (self.current.idx + 1) % len(self.pid_by_idx)
            self._remove_tracked(self.current)
            pid = self.pid_by_idx[idx]
            next_tracked = self.tracking[pid]
            next_tracked.client.flag = True
            self.current = next_tracked
            return False
        return None

    def _remove_tracked(self, tracked):
        """
            Remove tracked clients
        """
        del self.pid_by_idx[tracked.idx]
        del self.tracking[tracked.pid]
        del tracked


class ScheduleClient:
    def __init__(self, process, manager):
        super().__init__()

        self.process = process
        self.pid = process.pid
        self.flag = False
        self.manager = manager
        self.queue = manager.queue
    
    def register(self):
        self.manager.register_new_process(self.process, self)

    def check_flag(self):
        if self.flag:
            self.manager.announce_flag(self.pid)
            return True
        return False


class Tracked:
    def __init__(self, process, client, idx, initial_value=0):
        super().__init__()

        self.client = client
        self.process = process
        self.pid = process.pid  # this isn't explicitly needed, but it is here in case identifying a "lost" object becomes an issue
        self.idx = idx
        self.counter = initial_value
        self.time_with_flag = 0

    def get_value(self):
        self.counter += 1
        return self.counter-1