import logging
import multiprocessing as mp
import signal

from tqdm import tqdm


class AbstractWorker:
    """
    Abstract class to define general format of worker
    """

    def __init__(self, job_queue, response_queue, **kwargs):
        self.job_queue = job_queue
        self.response_queue = response_queue

    def process(self, message):
        raise NotImplementedError

    def _run(self):
        while True:
            message = self.job_queue.get(block=True)
            if message is not None:
                idx, contents = message
                response = self.process(contents)
                self.response_queue.put((idx, response))
            else:
                self.response_queue.put(None)
                return True

    @classmethod
    def terminate_all(cls, threads, interrupt: bool = False):
        pass

    @classmethod
    def init_queue(cls, maxsize=0):
        raise NotImplementedError

    @staticmethod
    def receive_messages_list(response_queue, num_messages, num_workers, visualize=True, order=False):
        num_received = 0
        responses = [None] * num_messages if order else []
        num_finished = 0

        # generate visualization
        if visualize:
            bar = tqdm(total=num_messages)
        else:
            bar = None

        while True:
            resp = response_queue.get(block=True)

            # confirmation of terminated worker
            if resp is None:
                num_finished += 1

                # all workers are terminated
                if num_finished >= num_workers:
                    if bar is not None:
                        bar.close()
                    return responses

            # received response
            else:
                num_received += 1
                # return results in order
                if order:
                    responses[resp[0]] = resp[1]

                # return results as they come
                else:
                    responses.append(resp[1])

                # update visualization
                if bar is not None:
                    bar.update(1)

    @staticmethod
    def receive_messages_dict(response_queue, num_messages, num_workers, visualize=True):
        num_received = 0
        responses = {}
        num_finished = 0

        # generate visualization
        if visualize:
            bar = tqdm(total=num_messages)
        else:
            bar = None

        while True:
            resp = response_queue.get(block=True)

            # confirmation of terminated worker
            if resp is None:
                num_finished += 1

                # all workers are terminated
                if num_finished >= num_workers:
                    if bar is not None:
                        bar.close()
                    return responses

            # received response
            else:
                num_received += 1
                # return results in order
                responses[resp[0]] = resp[1]

                # update visualization
                if bar is not None:
                    bar.update(1)

    @staticmethod
    def receive_messages_generator(response_queue, num_messages, num_workers, visualize=True, order=False):
        order_idx = 0
        num_received = 0
        responses = [None] * num_messages if order else []
        num_finished = 0

        # generate visualization
        if visualize:
            bar = tqdm(total=num_messages)
        else:
            bar = None

        while True:
            resp = response_queue.get(block=True)

            # confirmation of terminated worker
            if resp is None:
                num_finished += 1

                # all workers are terminated
                if num_finished >= num_workers:
                    if bar is not None:
                        bar.close()
                        return

            # received response
            else:
                num_received += 1
                # return results in order
                if order:
                    responses[resp[0]] = resp[1]

                    # yield all results which are ready
                    while order_idx < len(responses) and responses[order_idx] is not None:
                        yield responses[order_idx]
                        order_idx += 1

                # return results as they come
                else:
                    yield resp[1]

                # update visualization
                if bar is not None:
                    bar.update(1)

    @classmethod
    def execute_job_generator(
        cls, messages, *, num_workers=1, visualize=True, order=False, response_queue_maxsize=0, **worker_args
    ):
        if len(messages) == 0:
            return
        elif num_workers == 0:
            yield from cls.execute_job_main_thread_generator(messages, visualize=visualize, **worker_args)

        job_queue = cls.init_queue()
        response_queue = cls.init_queue(maxsize=response_queue_maxsize)

        threads = []

        # create workers
        for _ in range(min(num_workers, len(messages))):
            worker = cls(job_queue, response_queue, **worker_args)
            worker.start()
            threads.append(worker)

        # send messages
        for i, m in enumerate(messages):
            job_queue.put((i, m))

        # send poison pills
        for _ in range(len(threads)):
            job_queue.put(None)

        interrupt = False
        try:
            # receive messages
            yield from AbstractWorker.receive_messages_generator(
                response_queue, len(messages), len(threads), visualize=visualize, order=order
            )
        except KeyboardInterrupt:
            logging.error("Keyboard interrupt received. Terminating...")
            interrupt = True
        finally:
            cls.terminate_all(threads, interrupt=interrupt)

    @classmethod
    def execute_job(
        cls, messages, *, num_workers=1, visualize=True, order=False, response_queue_maxsize=0, **worker_args
    ):
        if len(messages) == 0:
            return
        elif num_workers == 0:
            return [out for out in cls.execute_job_main_thread_generator(messages, visualize=visualize, **worker_args)]

        job_queue = cls.init_queue()
        response_queue = cls.init_queue(maxsize=response_queue_maxsize)

        threads = []

        # create workers
        for _ in range(min(num_workers, len(messages))):
            worker = cls(job_queue, response_queue, **worker_args)
            worker.start()
            threads.append(worker)

        # send messages
        if isinstance(messages, (list, tuple)):
            for i, m in enumerate(messages):
                job_queue.put((i, m))
        elif isinstance(messages, dict):
            for key, value in messages.items():
                job_queue.put((key, value))
        else:
            raise TypeError(f"No put method implemented for messages variable of type {type(messages)}.")

        # send poison pills
        for _ in range(len(threads)):
            job_queue.put(None)

        # receive messages
        interrupt = False
        try:
            if isinstance(messages, (list, tuple)):
                return AbstractWorker.receive_messages_list(
                    response_queue, len(messages), len(threads), visualize=visualize, order=order
                )
            elif isinstance(messages, dict):
                return AbstractWorker.receive_messages_dict(
                    response_queue, len(messages), len(threads), visualize=visualize
                )
            else:
                raise TypeError(
                    f"No receive_message method implemented for messages variable of type {type(messages)}."
                )
        except KeyboardInterrupt:
            logging.error("Keyboard interrupt received. Terminating...")
            interrupt = True
        finally:
            # terminate threads/processes
            cls.terminate_all(threads, interrupt=interrupt)

    @classmethod
    def execute_job_main_thread_generator(cls, messages, *, visualize=True, **worker_args):
        """
        Execute a job in the main thread

        :param messages: _description_
        :param visualize: _description_, defaults to True
        :yield: _description_
        """
        if len(messages) == 0:
            return

        worker = cls(None, None, **worker_args)

        if visualize:
            bar = tqdm(total=len(messages))
        else:
            bar = None

        for m in messages:
            yield worker.process(m)
            if bar is not None:
                bar.update(1)

        if bar is not None:
            bar.close()


class AbstractProcessWorker(AbstractWorker, mp.Process):
    """
    Multiprocessing implementation of worker
    """

    def __init__(self, job_queue, response_queue, **kwargs):
        mp.Process.__init__(self)
        AbstractWorker.__init__(self, job_queue, response_queue)

    def process(self, message):
        raise NotImplementedError

    def run(self) -> None:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        self._run()

    @classmethod
    def init_queue(cls, maxsize=0):
        return mp.Queue(maxsize=maxsize)

    @classmethod
    def terminate_all(cls, threads, interrupt: bool = False):
        for t in threads:
            if interrupt:
                t.terminate()
            t.join()
