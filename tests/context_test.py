import sys
import time
sys.path.append("..")
import ascend

def context_test_001():
    acl_json = "./acl.json"
    device_set = {7, 9}
    context = ascend.Context(device_set, acl_json)
    for ctx in context.context_dict.items():
        print(ctx)

    for ctx in context:
        print(ctx)

    del context


def context_test_002():
    device_set = {4, 5, 6, 7}
    context = ascend.Context(device_set)
    print(context)
    for ctx in context:
        print(ctx)

    del context

context_test_001()
context_test_002()
