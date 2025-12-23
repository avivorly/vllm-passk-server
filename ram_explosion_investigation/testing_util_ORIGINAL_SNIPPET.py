# ORIGINAL BUGGY CODE - testing_util.py lines 428-440
# The comment says "max memory is set to 4GB" but NO VALUE IS PASSED!

def run_test(sample, test=None, debug=False, timeout=6):
    """
    if test(generated_code) is not None it'll try to run the code.
    otherwise it'll just return an input and output pair.
    """
    signal.signal(signal.SIGALRM, timeout_handler)

    # Disable functionalities that can make destructive changes to the test.
    # max memory is set to 4GB          <-- THIS COMMENT IS A LIE!
    reliability_guard()                  # <-- BUG: NO MEMORY LIMIT PASSED!

    # ... rest of function

# The reliability_guard function DOES support memory limits:
def reliability_guard(maximum_memory_bytes=None):  # Default is None!
    if maximum_memory_bytes is not None:           # This never runs!
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
    # ... rest of function
