# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import StateAndReward_pb2 as StateAndReward__pb2


class acerServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetExplorationAction = channel.unary_unary(
                '/service.acerService/GetExplorationAction',
                request_serializer=StateAndReward__pb2.StateReward.SerializeToString,
                response_deserializer=StateAndReward__pb2.Action.FromString,
                )


class acerServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def GetExplorationAction(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_acerServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetExplorationAction': grpc.unary_unary_rpc_method_handler(
                    servicer.GetExplorationAction,
                    request_deserializer=StateAndReward__pb2.StateReward.FromString,
                    response_serializer=StateAndReward__pb2.Action.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'service.acerService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class acerService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetExplorationAction(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/service.acerService/GetExplorationAction',
            StateAndReward__pb2.StateReward.SerializeToString,
            StateAndReward__pb2.Action.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)