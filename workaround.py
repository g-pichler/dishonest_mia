import flwr
import grpc

from flwr.client.grpc_client.connection import \
    (contextmanager,
     GRPC_MAX_MESSAGE_LENGTH,
     Iterator,
     Tuple,
     Callable,
     ServerMessage,
     Queue,
     ClientMessage,
     Optional,
     log,
     INFO,
     on_channel_state_change,
     FlowerServiceStub,
     DEBUG,
     )


@contextmanager
def grpc_connection(
        server_address: str,
        max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
        root_certificates: Optional[bytes] = None,
) -> Iterator[Tuple[Callable[[], ServerMessage], Callable[[ClientMessage], None]]]:
    """Establish an insecure gRPC connection to a gRPC server.

    Parameters
    ----------
    server_address : str
        The IPv6 address of the server. If the Flower server runs on the same machine
        on port 8080, then `server_address` would be `"[::]:8080"`.
    max_message_length : int
        The maximum length of gRPC messages that can be exchanged with the Flower
        server. The default should be sufficient for most models. Users who train
        very large models might need to increase this value. Note that the Flower
        server needs to be started with the same value
        (see `flwr.server.start_server`), otherwise it will not know about the
        increased limit and block larger messages.
        (default: 536_870_912, this equals 512MB)
    root_certificates : Optional[bytes] (default: None)
        The PEM-encoded root certificates as a byte string. If provided, a secure
        connection using the certificates will be established to a SSL-enabled
        Flower server

    Returns
    -------
    receive, send : Callable, Callable

    Examples
    --------
    Establishing a SSL-enabled connection to the server:

    >>> from pathlib import Path
    >>> with grpc_connection(
    >>>     server_address,
    >>>     max_message_length=max_message_length,
    >>>     root_certificates=Path("/crts/root.pem").read_bytes(),
    >>> ) as conn:
    >>>     receive, send = conn
    >>>     server_message = receive()
    >>>     # do something here
    >>>     send(client_message)
    """
    # Possible options:
    # https://github.com/grpc/grpc/blob/v1.43.x/include/grpc/impl/codegen/grpc_types.h
    channel_options = [
        ("grpc.max_send_message_length", max_message_length),
        ("grpc.max_receive_message_length", max_message_length),
    ]

    if root_certificates is not None:
        ssl_channel_credentials = grpc.ssl_channel_credentials(root_certificates)
        channel = grpc.secure_channel(
            server_address, ssl_channel_credentials, options=channel_options
        )
        log(INFO, "Opened secure gRPC connection using certificates")
    else:
        channel = grpc.insecure_channel(server_address, options=channel_options)
        log(INFO, "Opened insecure gRPC connection (no certificates were passed)")

    channel.subscribe(on_channel_state_change)

    queue: Queue[ClientMessage] = Queue(  # pylint: disable=unsubscriptable-object
        maxsize=1
    )
    stub = FlowerServiceStub(channel)

    server_message_iterator: Iterator[ServerMessage] = stub.Join(
        iter(queue.get, None), wait_for_ready=True,
    )

    receive: Callable[[], ServerMessage] = lambda: next(server_message_iterator)
    send: Callable[[ClientMessage], None] = lambda msg: queue.put(msg, block=False)

    try:
        yield (receive, send)
    finally:
        # Make sure to have a final
        channel.close()
        log(DEBUG, "gRPC channel closed")


# Patch it in
flwr.client.app.grpc_connection = grpc_connection
