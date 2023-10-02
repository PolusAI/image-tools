"""Provides the ProcessManager class."""

import abc
import concurrent.futures
import logging
import multiprocessing
import os
import queue
import typing


class ProcessManager(concurrent.futures.Executor):
    """Manages processes and threads.

    The ProcessManager class is a singleton class that manages processes and
    threads. It is a subclass of concurrent.futures.Executor and can be used
    as such. It is also a context manager and can be used as such.

    Attributes:
        unique_manager: The unique ProcessManager instance.
        log_level: The logging level of the ProcessManager.
        debug: Whether to print debug messages.
        name: The name of the ProcessManager.
        main_logger: The logger of the ProcessManager.
        job_name: The name of the current process.
        job_logger: The logger of the current process.
        process_executor: The process pool executor.
        processes: The list of processes' futures.
        thread_executor: The thread pool executor.
        threads: The list of threads' futures.
        num_active_threads: The number of active threads.
        num_available_cpus: The number of available CPUs.
        num_processes: The maximum number of processes.
        threads_per_process: The number of threads per process that can be run.
        threads_per_request: The number of threads per request.
        running: Whether the ProcessManager is running.
        process_queue: The queue of processes.
        process_names: The queue of process names.
        process_ids: The queue of process IDs.
        thread_queue: The queue of threads.
        thread_names: The queue of thread names.
        thread_ids: The queue of thread IDs.
    """

    unique_manager = None

    log_level = getattr(logging, os.environ.get("PREADATOR_LOG_LEVEL", "WARNING"))
    debug = os.getenv("PREADATOR_DEBUG", "false").lower() in ("true", "1", "t")

    name = "ProcessManager"
    main_logger = logging.getLogger(name)
    main_logger.setLevel(log_level)

    job_name = None
    job_logger = None

    process_executor = None
    processes = None

    thread_executor = None
    threads = None
    num_active_threads = 0

    # os.sched_getaffinity(0) is not available on MacOS and Windows.
    num_cpus: int = os.cpu_count()  # type: ignore[assignment]
    num_processes: int = max(1, num_cpus // 2)
    threads_per_process: int = num_cpus // num_processes
    threads_per_request: int = threads_per_process

    running: bool = False

    process_queue = None
    process_names = None
    process_ids = None

    thread_queue = None
    thread_names = None
    thread_ids = None

    def __new__(
        cls,
        *args,  # noqa: ANN002, ARG003
        **kwargs,  # noqa: ANN003, ARG003
    ) -> "ProcessManager":
        """Creates a new ProcessManager instance if none exists.

        Otherwise, returns the existing instance.

        Returns:
            A unique ProcessManager instance.
        """
        if cls.unique_manager is None:
            cls.unique_manager = super().__new__(cls)
        return cls.unique_manager

    def __init__(
        self,
        name: typing.Optional[str] = None,
        process_names: typing.Optional[list[str]] = None,
        **init_args,  # noqa: ANN003
    ) -> None:
        """Initializes the process manager.

        Args:
            name: The name of the ProcessManager.
            process_names: The names of the processes.

            init_args: The arguments to pass to the initializer. The following
            arguments are supported:
                log_level: The logging level of the ProcessManager.
                num_processes: The maximum number of processes.
                threads_per_process: The number of threads per process that can be run.
                threads_per_request: The number of threads per request.

        """
        # Check init_args.
        for k, v in init_args.items():
            if not hasattr(self, k):
                msg = f"Invalid initializer argument: {k}={v}"
                raise ValueError(msg)

        # Make sure that the ProcessManager is not already running.
        if self.running:
            msg = "ProcessManager is already running. Try shutting it down first."
            raise RuntimeError(msg)

        # Change the name of the logger.
        if name is not None:
            self.name = name
            self.main_logger = logging.getLogger(name)
            self.main_logger.setLevel(self.log_level)

        # Start the Process Manager.
        self.running = True
        self.main_logger.debug(
            f"Starting ProcessManager with {self.num_processes} processes ...",
        )

        # Create the process name queue.
        if process_names is None:
            process_names = [
                f"{self.name}: Process {i + 1}" for i in range(self.num_processes)
            ]
        elif len(process_names) != self.num_processes:
            msg = f"Number of process names must be equal to {self.num_processes}."
            raise ValueError(msg)

        self.process_names = multiprocessing.Queue(self.num_processes)
        for p_name in process_names:
            self.process_names.put(p_name)  # type: ignore[attr-defined]

        # Create the process queue and populate it.
        self.process_queue = multiprocessing.Queue(self.num_processes)
        for _ in range(self.num_processes):
            self.process_queue.put(self.threads_per_process)  # type: ignore[attr-defined]  # noqa: E501

        # Create the process IDs queue.
        self.process_ids = multiprocessing.Queue()

        # Create the process executor.
        self.process_executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.num_processes,
        )
        self.processes = []  # type: ignore[var-annotated]

        # Create the thread executor.
        self.thread_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_cpus,
        )
        self.threads = []  # type: ignore[var-annotated]

        if self.running:
            num_threads = self.threads_per_process
        else:
            num_threads = self.num_cpus // self.threads_per_request

        self.main_logger.debug(
            f"Starting the thread pool with {num_threads} threads ...",
        )
        self.thread_queue = multiprocessing.Queue(num_threads)
        for _ in range(0, num_threads, self.threads_per_request):
            self.thread_queue.put(self.threads_per_request)  # type: ignore[attr-defined]  # noqa: E501
        self.num_active_threads = num_threads

        self._initializer(**init_args)

    def _initializer(
        self,
        **init_args,  # noqa: ANN003
    ) -> None:
        # Initialize the process.
        for k, v in init_args.items():
            setattr(self, k, v)

        self.job_name = self.process_names.get(timeout=0)  # type: ignore[attr-defined]
        self.job_logger = logging.getLogger(self.job_name)
        self.job_logger.setLevel(self.log_level)

        self.process_ids.put(os.getpid())  # type: ignore[attr-defined]
        self.main_logger.debug(
            f"Process {self.job_name} initialized with PID {os.getpid()}.",
        )

    def process(self, name: typing.Optional[str] = None) -> "ProcessLock":
        """Acquires a process lock."""
        if name is not None:
            self.job_name = f"{name}: "

        return ProcessLock(self.process_queue)  # type: ignore[arg-type]

    def thread(self) -> "ThreadLock":
        """Acquires a thread lock."""
        return ThreadLock(self.thread_queue)  # type: ignore[arg-type]

    def submit_process(
        self,
        fn,  # noqa: ANN001
        /,
        *args,  # noqa: ANN002
        **kwargs,  # noqa: ANN003
    ) -> concurrent.futures.Future:
        """Submits a callable to be executed with the given arguments.

        Schedules the callable to be executed as fn(*args, **kwargs) and returns
        a Future instance representing the execution of the callable.

        Returns:
            A Future representing the given call.
        """
        self.processes.append(
            self.process_executor.submit(fn, *args, **kwargs),  # type: ignore[union-attr]  # noqa: E501
        )
        return self.processes[-1]

    def submit_thread(
        self,
        fn,  # noqa: ANN001
        /,
        *args,  # noqa: ANN002
        **kwargs,  # noqa: ANN003
    ) -> concurrent.futures.Future:
        """Submits a callable to be executed with the given arguments.

        Schedules the callable to be executed as fn(*args, **kwargs) and returns
        a Future instance representing the execution of the callable.

        Returns:
            A Future representing the given call.
        """
        self.threads.append(
            self.thread_executor.submit(fn, *args, **kwargs),  # type: ignore[union-attr]  # noqa: E501
        )
        return self.threads[-1]

    def submit(
        self,
        fn,  # noqa: ANN001, ARG002
        /,
        *args,  # noqa: ANN002, ARG002
        **kwargs,  # noqa: ANN003, ARG002
    ) -> concurrent.futures.Future:
        """Use `submit_process` or `submit_thread` instead."""
        msg = "Use `submit_process` or `submit_thread` instead."
        raise NotImplementedError(msg)

    def join_processes(self, update_period: float = 30) -> None:
        """Waits for all processes to finish."""
        if len(self.processes) == 0:
            if not self.debug:
                self.main_logger.warning("No processes to join.")
            self.running = False
            return

        errors = []

        while True:
            done, not_done = concurrent.futures.wait(
                self.processes,
                timeout=update_period,
                return_when=concurrent.futures.FIRST_EXCEPTION,
            )

            for p in done:
                if (e := p.exception(timeout=0)) is not None:
                    self.process_executor.shutdown(cancel_futures=True)  # type: ignore[union-attr]  # noqa: E501
                    self.job_logger.error(e)  # type: ignore[union-attr]
                    errors.append(e)

            progress = 100 * len(done) / (len(done) + len(not_done))
            self.main_logger.info(f"Processes progress: {progress:6.2f}% ...")

            if len(not_done) == 0:
                self.main_logger.info("All processes finished.")
                break

        if len(errors) > 0:
            msg = (
                f"{len(errors)} errors occurred during process join. "
                f"The first is: {errors[0]}"
            )
            self.main_logger.error(msg)
            raise errors[0]

    def join_threads(  # noqa: PLR0912, C901
        self,
        update_period: float = 10,
    ) -> None:
        """Waits for all threads to finish."""
        if len(self.threads) == 0:
            if not self.debug:
                self.main_logger.warning("No threads to join.")
            self.running = False
            return

        errors = []

        while True:
            done, not_done = concurrent.futures.wait(
                self.threads,
                timeout=update_period,
                return_when=concurrent.futures.FIRST_EXCEPTION,
            )

            for t in done:
                if (e := t.exception(timeout=0)) is not None:
                    self.thread_executor.shutdown(cancel_futures=True)  # type: ignore[union-attr]  # noqa: E501
                    self.job_logger.error(e)  # type: ignore[union-attr]
                    errors.append(e)

            progress = 100 * len(done) / (len(done) + len(not_done))
            self.main_logger.info(f"Threads progress: {progress:6.2f}% ...")

            if len(not_done) == 0:
                self.main_logger.info("All threads finished.")
                break

            # Steal threads from available processes.
            if (
                self.num_active_threads < self.num_cpus
                and self.process_queue is not None
            ):
                try:
                    num_new_threads = 0
                    while self.process_queue.get(timeout=0):
                        for _ in range(
                            0,
                            self.threads_per_process,
                            self.threads_per_request,
                        ):
                            self.thread_queue.put(self.threads_per_request, timeout=0)
                            num_new_threads += self.threads_per_request

                except queue.Empty:
                    if num_new_threads > 0:
                        self.num_active_threads += num_new_threads
                        self.main_logger.info(
                            f"Stole {num_new_threads} threads from processes.",
                        )

        if len(errors) > 0:
            msg = (
                f"{len(errors)} errors occurred during thread join. "
                f"The first is: {errors[0]}"
            )
            self.main_logger.error(msg)
            raise errors[0]

    def shutdown(  # noqa: PLR0912 (too-many-branches (13 > 12))
        self,
        wait: bool = True,
        *,
        cancel_futures: bool = False,
    ) -> None:
        """Clean-up the resources associated with the Executor.

        It is safe to call this method several times. Otherwise, no other
        methods can be called after this one.

        Args:
            wait: If True then shutdown will not return until all running
                futures have finished executing and the resources used by the
                executor have been reclaimed.
            cancel_futures: If True then shutdown will cancel all pending
                futures. Futures that are completed or running will not be
                cancelled.
        """
        errors = []

        if self.thread_executor is not None:
            self.main_logger.debug("Shutting down the thread pool ...")
            self.thread_executor.shutdown(wait=wait, cancel_futures=cancel_futures)

            if self.threads is not None:
                for t in self.threads:
                    try:
                        if (e := t.exception(timeout=0)) is not None:
                            errors.append(e)
                    except concurrent.futures.TimeoutError:
                        pass

        if self.process_executor is not None:
            self.main_logger.debug("Shutting down the process pool ...")
            self.process_executor.shutdown(wait=wait, cancel_futures=cancel_futures)

            if self.processes is not None:
                for p in self.processes:
                    try:
                        if (e := p.exception(timeout=0)) is not None:
                            errors.append(e)
                    except concurrent.futures.TimeoutError:
                        pass

        self.join_threads()
        self.join_processes()
        self.running = False
        self.thread_executor = None
        self.process_executor = None

        if len(errors) > 0:
            msg = (
                f"{len(errors)} errors occurred during shutdown. "
                f"The first is: {errors[0]}"
            )
            self.main_logger.error(msg)
            raise errors[0]


class QueueLock(abc.ABC):
    """An abstract class for dealing with locks in processes and threads."""

    name = "QueueLock"
    logger = logging.getLogger(name)
    logger.setLevel(ProcessManager.log_level)

    def __init__(self, queue: multiprocessing.Queue) -> None:
        """Initializes the lock with the queue."""
        self.queue = queue
        self.count = 0

    def lock(self) -> None:
        """Acquires the lock in the queue."""
        self.logger.debug("Acquiring lock ...")
        self.count = self.queue.get()
        self.logger.debug("Lock acquired.")

    def __del__(self) -> None:
        """Releases the lock in the queue."""
        if self.count != 0:
            self.logger.debug("Releasing lock ...")
            self.release()
            self.logger.debug("Lock released.")

    @abc.abstractmethod
    def release(self) -> None:
        """Releases the lock in the queue."""
        pass

    def __enter__(self) -> "QueueLock":
        """Acquires the lock in the queue."""
        self.lock()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # noqa: ANN001
        """Releases the lock in the queue."""
        self.release()
        self.count = 0


class ThreadLock(QueueLock):
    """A lock for threads."""

    name = "ThreadLock"
    logger = logging.getLogger(name)
    logger.setLevel(ProcessManager.log_level)

    def release(self) -> None:
        """Releases the lock in the queue."""
        self.logger.debug(f"Releasing {self.count} locks ...")
        self.queue.put(self.count)


class ProcessLock(QueueLock):
    """A lock for processes."""

    name = "ProcessLock"
    logger = logging.getLogger(name)
    logger.setLevel(ProcessManager.log_level)

    def release(self) -> None:
        """Releases the lock in the queue."""
        # We may have stolen some threads from other processes, so we need to
        # return them.
        try:
            num_returned = 0
            while True:
                num_returned += ProcessManager.thread_queue.get(  # type: ignore[attr-defined]  # noqa: E501
                    timeout=0,
                )

        except queue.Empty:
            self.logger.debug(f"Returning {num_returned} threads to the queue ...")
            for _ in range(
                0,
                num_returned,
                ProcessManager.threads_per_process,
            ):
                self.queue.put(
                    ProcessManager.threads_per_process,
                    timeout=0,
                )

            ProcessManager.num_active_threads = ProcessManager.threads_per_process
            ProcessManager.thread_queue.put(  # type: ignore[attr-defined]
                ProcessManager.threads_per_request,
            )
            self.logger.debug(f"Returned {num_returned} threads to the queue.")
