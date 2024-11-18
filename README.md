# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

# Parallel check output

```console
(.venv) rachna_umesh@dhcp-vl2053-399 mod3-rachnaumesh % python3 project/parallel_check.py              
MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
/Users/rachna_umesh/Documents/workspace/mod3-rachnaumesh/minitorch/fast_ops.py 
(178)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/rachna_umesh/Documents/workspace/mod3-rachnaumesh/minitorch/fast_ops.py (178) 
------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                       | 
        out: Storage,                                                               | 
        out_shape: Shape,                                                           | 
        out_strides: Strides,                                                       | 
        in_storage: Storage,                                                        | 
        in_shape: Shape,                                                            | 
        in_strides: Strides,                                                        | 
    ) -> None:                                                                      | 
        # TODO: Implement for Task 3.1.                                             | 
                                                                                    | 
        stride_aligned = np.array_equal(out_shape, in_shape) and np.array_equal(    | 
            out_strides, in_strides                                                 | 
        )                                                                           | 
                                                                                    | 
        if stride_aligned:                                                          | 
            for i in prange(out.size):----------------------------------------------| #2
                out[i] = fn(in_storage[i])                                          | 
        else:                                                                       | 
            for i in prange(out.size):----------------------------------------------| #3
                in_index = np.zeros(len(in_shape), dtype=np.int32)------------------| #0
                out_index = np.zeros(len(out_shape), dtype=np.int32)----------------| #1
                to_index(i, out_shape, out_index)                                   | 
                broadcast_index(out_index, out_shape, in_shape, in_index)           | 
                in_pos = index_to_position(in_index, in_strides)                    | 
                out_pos = index_to_position(out_index, out_strides)                 | 
                out[out_pos] = fn(in_storage[in_pos])                               | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 4 parallel for-
loop(s) (originating from loops labelled: #2, #3, #0, #1).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--3 is a parallel loop
   +--0 --> rewritten as a serial loop
   +--1 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (parallel)
   +--1 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (serial)
   +--1 (serial)


 
Parallel region 0 (loop #3) had 0 loop(s) fused and 2 loop(s) serialized as part
 of the larger parallel loop (#3).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/rachna_umesh/Documents/workspace/mod3-rachnaumesh/minitorch/fast_ops.py 
(197) is hoisted out of the parallel loop labelled #3 (it will be performed 
before the loop is executed and reused inside the loop):
   Allocation:: in_index = np.zeros(len(in_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/rachna_umesh/Documents/workspace/mod3-rachnaumesh/minitorch/fast_ops.py 
(198) is hoisted out of the parallel loop labelled #3 (it will be performed 
before the loop is executed and reused inside the loop):
   Allocation:: out_index = np.zeros(len(out_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/Users/rachna_umesh/Documents/workspace/mod3-rachnaumesh/minitorch/fast_ops.py 
(231)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/rachna_umesh/Documents/workspace/mod3-rachnaumesh/minitorch/fast_ops.py (231) 
---------------------------------------------------------------------------|loop #ID
    def _zip(                                                              | 
        out: Storage,                                                      | 
        out_shape: Shape,                                                  | 
        out_strides: Strides,                                              | 
        a_storage: Storage,                                                | 
        a_shape: Shape,                                                    | 
        a_strides: Strides,                                                | 
        b_storage: Storage,                                                | 
        b_shape: Shape,                                                    | 
        b_strides: Strides,                                                | 
    ) -> None:                                                             | 
        # TODO: Implement for Task 3.1.                                    | 
        stride_aligned = (                                                 | 
            np.array_equal(out_shape, a_shape)                             | 
            and np.array_equal(out_shape, b_shape)                         | 
            and np.array_equal(out_strides, a_strides)                     | 
            and np.array_equal(out_strides, b_strides)                     | 
        )                                                                  | 
                                                                           | 
        if stride_aligned:                                                 | 
            for i in prange(out.size):-------------------------------------| #7
                out[i] = fn(a_storage[i], b_storage[i])                    | 
        else:                                                              | 
            for i in prange(out.size):-------------------------------------| #8
                a_index = np.zeros(len(a_shape), dtype=np.int32)-----------| #4
                b_index = np.zeros(len(b_shape), dtype=np.int32)-----------| #5
                out_index = np.zeros(len(out_shape), dtype=np.int32)-------| #6
                to_index(i, out_shape, out_index)                          | 
                broadcast_index(out_index, out_shape, a_shape, a_index)    | 
                broadcast_index(out_index, out_shape, b_shape, b_index)    | 
                a_pos = index_to_position(a_index, a_strides)              | 
                b_pos = index_to_position(b_index, b_strides)              | 
                out_pos = index_to_position(out_index, out_strides)        | 
                out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])      | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 5 parallel for-
loop(s) (originating from loops labelled: #7, #8, #4, #5, #6).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--8 is a parallel loop
   +--4 --> rewritten as a serial loop
   +--5 --> rewritten as a serial loop
   +--6 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (parallel)
   +--5 (parallel)
   +--6 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (serial)
   +--5 (serial)
   +--6 (serial)


 
Parallel region 0 (loop #8) had 0 loop(s) fused and 3 loop(s) serialized as part
 of the larger parallel loop (#8).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/rachna_umesh/Documents/workspace/mod3-rachnaumesh/minitorch/fast_ops.py 
(255) is hoisted out of the parallel loop labelled #8 (it will be performed 
before the loop is executed and reused inside the loop):
   Allocation:: a_index = np.zeros(len(a_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/rachna_umesh/Documents/workspace/mod3-rachnaumesh/minitorch/fast_ops.py 
(256) is hoisted out of the parallel loop labelled #8 (it will be performed 
before the loop is executed and reused inside the loop):
   Allocation:: b_index = np.zeros(len(b_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/rachna_umesh/Documents/workspace/mod3-rachnaumesh/minitorch/fast_ops.py 
(257) is hoisted out of the parallel loop labelled #8 (it will be performed 
before the loop is executed and reused inside the loop):
   Allocation:: out_index = np.zeros(len(out_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/Users/rachna_umesh/Documents/workspace/mod3-rachnaumesh/minitorch/fast_ops.py 
(290)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/rachna_umesh/Documents/workspace/mod3-rachnaumesh/minitorch/fast_ops.py (290) 
--------------------------------------------------------------------|loop #ID
    def _reduce(                                                    | 
        out: Storage,                                               | 
        out_shape: Shape,                                           | 
        out_strides: Strides,                                       | 
        a_storage: Storage,                                         | 
        a_shape: Shape,                                             | 
        a_strides: Strides,                                         | 
        reduce_dim: int,                                            | 
    ) -> None:                                                      | 
        # TODO: Implement for Task 3.1.                             | 
        reduce_size = a_shape[reduce_dim]                           | 
        for i in prange(len(out)):----------------------------------| #10
            out_index = np.zeros(len(out_shape), dtype=np.int32)----| #9
            to_index(i, out_shape, out_index)                       | 
            o = index_to_position(out_index, out_strides)           | 
            acc = out[o]  # Use accumulator variable                | 
            for s in range(reduce_size):                            | 
                out_index[reduce_dim] = s                           | 
                j = index_to_position(out_index, a_strides)         | 
                acc = fn(acc, a_storage[j])                         | 
            out[o] = acc                                            | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #10, #9).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--10 is a parallel loop
   +--9 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--9 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--9 (serial)


 
Parallel region 0 (loop #10) had 0 loop(s) fused and 1 loop(s) serialized as 
part of the larger parallel loop (#10).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/rachna_umesh/Documents/workspace/mod3-rachnaumesh/minitorch/fast_ops.py 
(302) is hoisted out of the parallel loop labelled #10 (it will be performed 
before the loop is executed and reused inside the loop):
   Allocation:: out_index = np.zeros(len(out_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
/Users/rachna_umesh/Documents/workspace/mod3-rachnaumesh/minitorch/fast_ops.py 
(315)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/rachna_umesh/Documents/workspace/mod3-rachnaumesh/minitorch/fast_ops.py (315) 
------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                              | 
    out: Storage,                                                                         | 
    out_shape: Shape,                                                                     | 
    out_strides: Strides,                                                                 | 
    a_storage: Storage,                                                                   | 
    a_shape: Shape,                                                                       | 
    a_strides: Strides,                                                                   | 
    b_storage: Storage,                                                                   | 
    b_shape: Shape,                                                                       | 
    b_strides: Strides,                                                                   | 
) -> None:                                                                                | 
    """NUMBA tensor matrix multiply function.                                             | 
                                                                                          | 
    Should work for any tensor shapes that broadcast as long as                           | 
                                                                                          | 
    ```                                                                                   | 
    assert a_shape[-1] == b_shape[-2]                                                     | 
    ```                                                                                   | 
                                                                                          | 
    Optimizations:                                                                        | 
                                                                                          | 
    * Outer loop in parallel                                                              | 
    * No index buffers or function calls                                                  | 
    * Inner loop should have no global writes, 1 multiply.                                | 
                                                                                          | 
                                                                                          | 
    Args:                                                                                 | 
    ----                                                                                  | 
        out (Storage): storage for `out` tensor                                           | 
        out_shape (Shape): shape for `out` tensor                                         | 
        out_strides (Strides): strides for `out` tensor                                   | 
        a_storage (Storage): storage for `a` tensor                                       | 
        a_shape (Shape): shape for `a` tensor                                             | 
        a_strides (Strides): strides for `a` tensor                                       | 
        b_storage (Storage): storage for `b` tensor                                       | 
        b_shape (Shape): shape for `b` tensor                                             | 
        b_strides (Strides): strides for `b` tensor                                       | 
                                                                                          | 
    Returns:                                                                              | 
    -------                                                                               | 
        None : Fills in `out`                                                             | 
                                                                                          | 
    """                                                                                   | 
    assert a_shape[-1] == b_shape[-2]                                                     | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                | 
                                                                                          | 
    # TODO: Implement for Task 3.2.                                                       | 
    for i in prange(a_shape[0]):----------------------------------------------------------| #11
        for j in range(a_shape[1]):                                                       | 
            for k in range(b_shape[2]):                                                   | 
                sum = 0.0                                                                 | 
                for l in range(a_shape[-1]):                                              | 
                    a_pos = i * a_batch_stride + j * a_strides[1] + l * a_strides[2]      | 
                    b_pos = i * b_batch_stride + l * b_strides[1] + k * b_strides[2]      | 
                    sum += a_storage[a_pos] * b_storage[b_pos]                            | 
                out_pos = i * out_strides[0] + j * out_strides[1] + k * out_strides[2]    | 
                out[out_pos] = sum                                                        | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #11).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```

# Training logs and Epoch time for Small Models
# 1. Simple Dataset :

```bash
PYTHONPATH='/content/$DIR'
!python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.05
```

## CPU output:
Epoch  0  loss  6.453734794473642 correct 42
Epoch  10  loss  1.8045398383349154 correct 50
Epoch  20  loss  1.8333374431311014 correct 49
Epoch  30  loss  1.2202414603203562 correct 50
Epoch  40  loss  0.7241152661200221 correct 50
Epoch  50  loss  0.48623236432941247 correct 50
Epoch  60  loss  0.866109027464783 correct 50
Epoch  70  loss  0.8196562274396344 correct 50
Epoch  80  loss  0.3743489385105666 correct 50
Epoch  90  loss  0.7226238217395755 correct 50
Epoch  100  loss  0.346847347144442 correct 50
Epoch  110  loss  1.1769116266397668 correct 50
Epoch  120  loss  0.9420798866002568 correct 50
Epoch  130  loss  0.24466148057888235 correct 50
Epoch  140  loss  0.049691171099175646 correct 50
Epoch  150  loss  0.04760979959577983 correct 50
Epoch  160  loss  0.5228068590126844 correct 50
Epoch  170  loss  0.4741018780272215 correct 50
Epoch  180  loss  0.10070628083712024 correct 50
Epoch  190  loss  0.26139830788465196 correct 50
Epoch  200  loss  0.07725304708452825 correct 50
Epoch  210  loss  0.07139251166633725 correct 50
Epoch  220  loss  0.11606961726406996 correct 50
Epoch  230  loss  0.024357729689762863 correct 50
Epoch  240  loss  0.25924333344544537 correct 50
Epoch  250  loss  0.01452529981172195 correct 50
Epoch  260  loss  0.001621123085418088 correct 50
Epoch  270  loss  0.00416948691989441 correct 50
Epoch  280  loss  0.22129619202334969 correct 50
Epoch  290  loss  0.10693081796927624 correct 50
Epoch  300  loss  0.26107076402660195 correct 50
Epoch  310  loss  0.003647731869856227 correct 50
Epoch  320  loss  0.33168814229769783 correct 50
Epoch  330  loss  0.10242275769653877 correct 50
Epoch  340  loss  0.06998788289222714 correct 50
Epoch  350  loss  0.2841485912985119 correct 50
Epoch  360  loss  0.04078864011737887 correct 50
Epoch  370  loss  0.01452224169213017 correct 50
Epoch  380  loss  0.01540670019868307 correct 50
Epoch  390  loss  0.03543376149010665 correct 50
Epoch  400  loss  0.27661995640862735 correct 50
Epoch  410  loss  0.024008549450572916 correct 50
Epoch  420  loss  0.07535738227289698 correct 50
Epoch  430  loss  0.2776944088230062 correct 50
Epoch  440  loss  0.1575884788780757 correct 50
Epoch  450  loss  0.18026478416178776 correct 50
Epoch  460  loss  0.026596190688983654 correct 50
Epoch  470  loss  0.06984232143779881 correct 50
Epoch  480  loss  0.00041418452922751083 correct 50
Epoch  490  loss  0.039722226785217814 correct 50
Average epoch time: 0.1306s

## GPU output :
Epoch  0  loss  5.46260130206641 correct 40
Epoch  10  loss  2.4136447179704685 correct 47
Epoch  20  loss  1.2037462958252783 correct 48
Epoch  30  loss  1.1486052056593938 correct 49
Epoch  40  loss  1.1258483539654527 correct 49
Epoch  50  loss  0.9777549736215893 correct 50
Epoch  60  loss  0.146664468463481 correct 50
Epoch  70  loss  1.0023791129646025 correct 49
Epoch  80  loss  0.22479091522392572 correct 49
Epoch  90  loss  0.5695492306161417 correct 49
Epoch  100  loss  0.6527792838237754 correct 50
Epoch  110  loss  0.15943290952980785 correct 49
Epoch  120  loss  0.9476346215255946 correct 50
Epoch  130  loss  0.8051084663067443 correct 49
Epoch  140  loss  0.13971325388607198 correct 49
Epoch  150  loss  0.038452618061264376 correct 49
Epoch  160  loss  0.04125476901390151 correct 50
Epoch  170  loss  0.01529843008493288 correct 49
Epoch  180  loss  0.2428425097366106 correct 49
Epoch  190  loss  0.5170501979009414 correct 50
Epoch  200  loss  0.6098385875978264 correct 49
Epoch  210  loss  0.5513204650658645 correct 50
Epoch  220  loss  0.03251074194220931 correct 50
Epoch  230  loss  0.6681577493221189 correct 50
Epoch  240  loss  0.1882185489539317 correct 49
Epoch  250  loss  0.2813332021488674 correct 50
Epoch  260  loss  0.655264189777376 correct 50
Epoch  270  loss  0.001936426865625034 correct 49
Epoch  280  loss  0.0005983661262669466 correct 50
Epoch  290  loss  0.577324749712921 correct 50
Epoch  300  loss  0.7379700328934192 correct 50
Epoch  310  loss  0.03452268604584894 correct 50
Epoch  320  loss  0.527712320210049 correct 50
Epoch  330  loss  0.007840902447246922 correct 50
Epoch  340  loss  0.5233548736811149 correct 50
Epoch  350  loss  0.049641003127444246 correct 50
Epoch  360  loss  0.6215675509587169 correct 50
Epoch  370  loss  0.03644453305806447 correct 50
Epoch  380  loss  0.7170566108997849 correct 50
Epoch  390  loss  0.061395632043946866 correct 50
Epoch  400  loss  0.41960709159274334 correct 50
Epoch  410  loss  0.2762682791794102 correct 50
Epoch  420  loss  0.03830469485039724 correct 50
Epoch  430  loss  0.005227032035232054 correct 50
Epoch  440  loss  0.6988400651996589 correct 50
Epoch  450  loss  0.014427859669447783 correct 50
Epoch  460  loss  0.04225785539997603 correct 50
Epoch  470  loss  0.4562270316129772 correct 50
Epoch  480  loss  0.007935689937363585 correct 50
Epoch  490  loss  0.004599829681376358 correct 50
Average epoch time: 1.2208s

# 2. Split Dataset :

```bash
PYTHONPATH='/content/$DIR'
!python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05
```

## CPU output:
Epoch  0  loss  7.490674900984996 correct 31
Epoch  10  loss  7.101848436430069 correct 40
Epoch  20  loss  6.536716767206154 correct 35
Epoch  30  loss  4.000598468310852 correct 42
Epoch  40  loss  3.3204919894342213 correct 46
Epoch  50  loss  3.9254219663872325 correct 47
Epoch  60  loss  2.120861279338868 correct 48
Epoch  70  loss  2.1503907606909918 correct 48
Epoch  80  loss  1.6566593731918688 correct 48
Epoch  90  loss  2.076573732933265 correct 49
Epoch  100  loss  2.767439688363944 correct 45
Epoch  110  loss  3.633452245031412 correct 49
Epoch  120  loss  1.3487484356334267 correct 50
Epoch  130  loss  1.9158392732647787 correct 48
Epoch  140  loss  1.5041129308020946 correct 50
Epoch  150  loss  1.6452964508426602 correct 49
Epoch  160  loss  0.8334460916587143 correct 50
Epoch  170  loss  1.9275850345280658 correct 48
Epoch  180  loss  1.342552347360858 correct 49
Epoch  190  loss  1.198032873835351 correct 50
Epoch  200  loss  0.47695052786799474 correct 50
Epoch  210  loss  0.8160953602241512 correct 50
Epoch  220  loss  1.9407663685306564 correct 49
Epoch  230  loss  0.6378632944814211 correct 50
Epoch  240  loss  0.20380286836550862 correct 50
Epoch  250  loss  0.14616972870632114 correct 49
Epoch  260  loss  0.8086913718312303 correct 50
Epoch  270  loss  0.8208034163044174 correct 49
Epoch  280  loss  0.10305602093271317 correct 49
Epoch  290  loss  0.11015608424802967 correct 50
Epoch  300  loss  0.8041747824036996 correct 50
Epoch  310  loss  0.6738992636217731 correct 50
Epoch  320  loss  1.063058596889516 correct 50
Epoch  330  loss  1.0495845310994683 correct 50
Epoch  340  loss  0.5511953618594181 correct 50
Epoch  350  loss  0.32051684514656065 correct 49
Epoch  360  loss  0.29772087730823116 correct 49
Epoch  370  loss  0.17839377326895559 correct 50
Epoch  380  loss  0.28219551313473856 correct 50
Epoch  390  loss  0.7296654592033137 correct 50
Epoch  400  loss  0.5900552884898216 correct 50
Epoch  410  loss  0.3270639845827572 correct 50
Epoch  420  loss  0.09041516151472537 correct 50
Epoch  430  loss  0.634531325256996 correct 50
Epoch  440  loss  0.34495436198351326 correct 50
Epoch  450  loss  0.852924727833895 correct 49
Epoch  460  loss  0.4625347140618939 correct 50
Epoch  470  loss  0.3065806654127446 correct 50
Epoch  480  loss  0.38592302809380613 correct 49
Epoch  490  loss  0.7739005813177762 correct 50
Average epoch time: 0.1284s

## Gpu output:
Epoch  0  loss  5.6435434153930135 correct 34
Epoch  10  loss  3.9050184178503113 correct 36
Epoch  20  loss  6.670290215014271 correct 38
Epoch  30  loss  6.918820499223709 correct 39
Epoch  40  loss  4.708863132056186 correct 39
Epoch  50  loss  4.035957843730745 correct 44
Epoch  60  loss  4.537948699966906 correct 44
Epoch  70  loss  3.2509166557076488 correct 46
Epoch  80  loss  1.4717983832760764 correct 38
Epoch  90  loss  2.5196627538242566 correct 45
Epoch  100  loss  2.4521065973509986 correct 46
Epoch  110  loss  1.5628539093357374 correct 46
Epoch  120  loss  4.176921417454285 correct 44
Epoch  130  loss  3.7320284998293394 correct 42
Epoch  140  loss  2.398881944748385 correct 48
Epoch  150  loss  0.9952544700064988 correct 43
Epoch  160  loss  2.566959305630398 correct 45
Epoch  170  loss  1.7712320875260126 correct 48
Epoch  180  loss  2.644293047179106 correct 49
Epoch  190  loss  1.2475308159237013 correct 49
Epoch  200  loss  5.665563715199605 correct 49
Epoch  210  loss  2.448594754362711 correct 47
Epoch  220  loss  2.7145553513648295 correct 44
Epoch  230  loss  0.7272600003163718 correct 49
Epoch  240  loss  0.9186990853918251 correct 44
Epoch  250  loss  1.6840241933067497 correct 48
Epoch  260  loss  1.3533996568239357 correct 47
Epoch  270  loss  1.46215070081972 correct 49
Epoch  280  loss  1.01272885677471 correct 46
Epoch  290  loss  1.0235823672620623 correct 44
Epoch  300  loss  2.3811907310278713 correct 44
Epoch  310  loss  2.7287215934740727 correct 48
Epoch  320  loss  2.8620344704704754 correct 44
Epoch  330  loss  1.5376939426439744 correct 48
Epoch  340  loss  0.361973312950091 correct 48
Epoch  350  loss  0.8137170692837035 correct 50
Epoch  360  loss  0.8132041184799248 correct 50
Epoch  370  loss  1.5444724286690303 correct 50
Epoch  380  loss  0.7130445760680353 correct 48
Epoch  390  loss  0.6896558458632459 correct 49
Epoch  400  loss  1.326226140759459 correct 50
Epoch  410  loss  1.342902427833232 correct 49
Epoch  420  loss  0.849300205460835 correct 46
Epoch  430  loss  1.2249421766900461 correct 50
Epoch  440  loss  0.8179710585021737 correct 50
Epoch  450  loss  0.9253325838088948 correct 50
Epoch  460  loss  4.512586668256595 correct 45
Epoch  470  loss  1.447825935713918 correct 45
Epoch  480  loss  0.0478102776805102 correct 48
Epoch  490  loss  0.8274387748728088 correct 50
Average epoch time: 1.1975s


# 1. XOR Dataset :

```bash
PYTHONPATH='/content/$DIR'
!python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.05
```

## CPU output:
Epoch  0  loss  8.571475935243257 correct 31
Epoch  10  loss  5.9621913386159004 correct 26
Epoch  20  loss  6.32193261998588 correct 40
Epoch  30  loss  4.597266614327768 correct 44
Epoch  40  loss  3.3876098283802656 correct 45
Epoch  50  loss  2.9078706511205663 correct 45
Epoch  60  loss  3.7799184886157198 correct 45
Epoch  70  loss  3.02519076011992 correct 44
Epoch  80  loss  4.313748648906008 correct 46
Epoch  90  loss  1.3886803110498704 correct 41
Epoch  100  loss  4.185465621549911 correct 43
Epoch  110  loss  6.205978276456549 correct 46
Epoch  120  loss  2.9969136312731686 correct 45
Epoch  130  loss  2.7166539675415917 correct 45
Epoch  140  loss  2.4492452636278506 correct 45
Epoch  150  loss  2.0512393761303445 correct 46
Epoch  160  loss  1.0596244086837525 correct 47
Epoch  170  loss  1.474523113900749 correct 45
Epoch  180  loss  4.672706651602166 correct 40
Epoch  190  loss  1.3554050193285787 correct 47
Epoch  200  loss  1.8488026738258652 correct 48
Epoch  210  loss  3.6685346439186066 correct 48
Epoch  220  loss  1.4429150808789049 correct 48
Epoch  230  loss  2.2171805190634344 correct 49
Epoch  240  loss  1.1805571790009552 correct 48
Epoch  250  loss  4.105894119178558 correct 43
Epoch  260  loss  3.9976902054722716 correct 46
Epoch  270  loss  2.705918994413283 correct 43
Epoch  280  loss  0.8738993417500265 correct 49
Epoch  290  loss  1.6867531969658398 correct 49
Epoch  300  loss  0.9226932749007373 correct 48
Epoch  310  loss  2.7931840508085557 correct 49
Epoch  320  loss  0.8833805804428845 correct 46
Epoch  330  loss  1.2466250993902732 correct 49
Epoch  340  loss  0.5244340796432612 correct 48
Epoch  350  loss  1.5742370228042069 correct 48
Epoch  360  loss  0.3737784707036123 correct 49
Epoch  370  loss  1.8133626045550806 correct 48
Epoch  380  loss  1.1498096121930612 correct 48
Epoch  390  loss  0.9200909178212134 correct 48
Epoch  400  loss  0.9476258282128646 correct 48
Epoch  410  loss  1.5238878151483128 correct 50
Epoch  420  loss  1.529973269695761 correct 47
Epoch  430  loss  1.2410178229506077 correct 48
Epoch  440  loss  0.9835710413731831 correct 49
Epoch  450  loss  1.2197022428194628 correct 49
Epoch  460  loss  1.1761390692721765 correct 49
Epoch  470  loss  0.881461593265054 correct 49
Epoch  480  loss  0.865583682647653 correct 48
Epoch  490  loss  1.3494175304268565 correct 49
Average epoch time: 0.1255s

## GPU output:
Epoch  0  loss  6.982799013700875 correct 42
Epoch  10  loss  6.0407462959863665 correct 42
Epoch  20  loss  2.133069331810441 correct 47
Epoch  30  loss  3.4378791771891226 correct 49
Epoch  40  loss  2.0684166118997287 correct 49
Epoch  50  loss  1.3462850437274476 correct 49
Epoch  60  loss  0.7860962368411988 correct 49
Epoch  70  loss  1.0876563477635042 correct 49
Epoch  80  loss  0.9031395661732091 correct 50
Epoch  90  loss  1.00497458430609 correct 50
Epoch  100  loss  1.0009286331501657 correct 50
Epoch  110  loss  0.7500235075309049 correct 50
Epoch  120  loss  0.9528382777735627 correct 50
Epoch  130  loss  0.6384819656546585 correct 50
Epoch  140  loss  0.6075939896641602 correct 50
Epoch  150  loss  0.6253487755047584 correct 50
Epoch  160  loss  0.8648213677688192 correct 50
Epoch  170  loss  0.6043818469010143 correct 50
Epoch  180  loss  0.6881756918361552 correct 50
Epoch  190  loss  0.5834262866191953 correct 50
Epoch  200  loss  0.38492146771217756 correct 50
Epoch  210  loss  0.9231608113048089 correct 50
Epoch  220  loss  0.20271766567644073 correct 50
Epoch  230  loss  0.3512390547404266 correct 50
Epoch  240  loss  0.842824517970001 correct 50
Epoch  250  loss  0.8583661249377635 correct 50
Epoch  260  loss  0.1636086888477521 correct 50
Epoch  270  loss  0.0643632698533898 correct 50
Epoch  280  loss  0.36459451219731565 correct 50
Epoch  290  loss  0.4318822516970415 correct 50
Epoch  300  loss  0.5365652895980382 correct 50
Epoch  310  loss  0.38263568668601516 correct 50
Epoch  320  loss  0.22093112433251508 correct 50
Epoch  330  loss  0.3040549446058446 correct 50
Epoch  340  loss  0.24430112462428455 correct 50
Epoch  350  loss  0.12566790058283459 correct 50
Epoch  360  loss  0.19095808281197163 correct 50
Epoch  370  loss  0.30080417774077056 correct 50
Epoch  380  loss  0.1343785062762729 correct 50
Epoch  390  loss  0.2686607768990022 correct 50
Epoch  400  loss  0.059221252258103665 correct 50
Epoch  410  loss  0.10562718034622375 correct 50
Epoch  420  loss  0.10189378622876652 correct 50
Epoch  430  loss  0.14869796139202787 correct 50
Epoch  440  loss  0.208883789404285 correct 50
Epoch  450  loss  0.138030120894785 correct 50
Epoch  460  loss  0.0822552723879125 correct 50
Epoch  470  loss  0.5285055753624606 correct 50
Epoch  480  loss  0.13673311757846843 correct 50
Epoch  490  loss  0.1638154917068988 correct 50
Average epoch time: 1.2029s

# Training logs and Epoch time for Large Model

