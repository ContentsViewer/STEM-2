��
��
B
AddV2
x"T
y"T
z"T"
Ttype:
2	��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
:
Maximum
x"T
y"T
z"T"
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
=
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
3
Square
x"T
y"T"
Ttype:
2
	
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
�
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type
output_handle"
element_dtypetype"

shape_typetype:
2	
�
TensorListReserve
element_shape"
shape_type
num_elements

handle"
element_dtypetype"

shape_typetype:
2	
�
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint���������
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
�
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
�"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8��
l
Adagrad/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameAdagrad/iter
e
 Adagrad/iter/Read/ReadVariableOpReadVariableOpAdagrad/iter*
_output_shapes
: *
dtype0	
n
Adagrad/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdagrad/decay
g
!Adagrad/decay/Read/ReadVariableOpReadVariableOpAdagrad/decay*
_output_shapes
: *
dtype0
~
Adagrad/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdagrad/learning_rate
w
)Adagrad/learning_rate/Read/ReadVariableOpReadVariableOpAdagrad/learning_rate*
_output_shapes
: *
dtype0
�
LSTM/lstm/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*+
shared_nameLSTM/lstm/lstm_cell/kernel
�
.LSTM/lstm/lstm_cell/kernel/Read/ReadVariableOpReadVariableOpLSTM/lstm/lstm_cell/kernel*
_output_shapes
:	�*
dtype0
�
$LSTM/lstm/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*5
shared_name&$LSTM/lstm/lstm_cell/recurrent_kernel
�
8LSTM/lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp$LSTM/lstm/lstm_cell/recurrent_kernel* 
_output_shapes
:
��*
dtype0
�
LSTM/lstm/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_nameLSTM/lstm/lstm_cell/bias
�
,LSTM/lstm/lstm_cell/bias/Read/ReadVariableOpReadVariableOpLSTM/lstm/lstm_cell/bias*
_output_shapes	
:�*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
�
.Adagrad/LSTM/lstm/lstm_cell/kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*?
shared_name0.Adagrad/LSTM/lstm/lstm_cell/kernel/accumulator
�
BAdagrad/LSTM/lstm/lstm_cell/kernel/accumulator/Read/ReadVariableOpReadVariableOp.Adagrad/LSTM/lstm/lstm_cell/kernel/accumulator*
_output_shapes
:	�*
dtype0
�
8Adagrad/LSTM/lstm/lstm_cell/recurrent_kernel/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*I
shared_name:8Adagrad/LSTM/lstm/lstm_cell/recurrent_kernel/accumulator
�
LAdagrad/LSTM/lstm/lstm_cell/recurrent_kernel/accumulator/Read/ReadVariableOpReadVariableOp8Adagrad/LSTM/lstm/lstm_cell/recurrent_kernel/accumulator* 
_output_shapes
:
��*
dtype0
�
,Adagrad/LSTM/lstm/lstm_cell/bias/accumulatorVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*=
shared_name.,Adagrad/LSTM/lstm/lstm_cell/bias/accumulator
�
@Adagrad/LSTM/lstm/lstm_cell/bias/accumulator/Read/ReadVariableOpReadVariableOp,Adagrad/LSTM/lstm/lstm_cell/bias/accumulator*
_output_shapes	
:�*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
layer-0
layer_with_weights-0
layer-1
layer-2
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api
	
signatures
 
\

lstm
trainable_variables
regularization_losses
	variables
	keras_api

	keras_api
a
iter
	decay
learning_rateaccumulator:accumulator;accumulator<

0
1
2
 

0
1
2
�
metrics
trainable_variables
non_trainable_variables
layer_metrics
regularization_losses
	variables

layers
layer_regularization_losses
 
l
cell

state_spec
trainable_variables
regularization_losses
	variables
 	keras_api

0
1
2
 

0
1
2
�
!metrics
trainable_variables
"non_trainable_variables
#layer_metrics
regularization_losses
	variables

$layers
%layer_regularization_losses
 
KI
VARIABLE_VALUEAdagrad/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEAdagrad/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEAdagrad/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUELSTM/lstm/lstm_cell/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE$LSTM/lstm/lstm_cell/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUELSTM/lstm/lstm_cell/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE

&0
 
 

0
1
2
 
~

kernel
recurrent_kernel
bias
'trainable_variables
(regularization_losses
)	variables
*	keras_api
 

0
1
2
 

0
1
2
�

+states
,metrics
trainable_variables
-non_trainable_variables
.layer_metrics
regularization_losses
	variables

/layers
0layer_regularization_losses
 
 
 


0
 
4
	1total
	2count
3	variables
4	keras_api

0
1
2
 

0
1
2
�
5metrics
6non_trainable_variables
7layer_metrics
'trainable_variables
(regularization_losses
)	variables

8layers
9layer_regularization_losses
 
 
 
 

0
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

10
21

3	variables
 
 
 
 
 
��
VARIABLE_VALUE.Adagrad/LSTM/lstm/lstm_cell/kernel/accumulatorVtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE8Adagrad/LSTM/lstm/lstm_cell/recurrent_kernel/accumulatorVtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE,Adagrad/LSTM/lstm/lstm_cell/bias/accumulatorVtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_input_1Placeholder*+
_output_shapes
:���������d*
dtype0* 
shape:���������d
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1LSTM/lstm/lstm_cell/kernel$LSTM/lstm/lstm_cell/recurrent_kernelLSTM/lstm/lstm_cell/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_5288575
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename Adagrad/iter/Read/ReadVariableOp!Adagrad/decay/Read/ReadVariableOp)Adagrad/learning_rate/Read/ReadVariableOp.LSTM/lstm/lstm_cell/kernel/Read/ReadVariableOp8LSTM/lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOp,LSTM/lstm/lstm_cell/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpBAdagrad/LSTM/lstm/lstm_cell/kernel/accumulator/Read/ReadVariableOpLAdagrad/LSTM/lstm/lstm_cell/recurrent_kernel/accumulator/Read/ReadVariableOp@Adagrad/LSTM/lstm/lstm_cell/bias/accumulator/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_save_5290235
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameAdagrad/iterAdagrad/decayAdagrad/learning_rateLSTM/lstm/lstm_cell/kernel$LSTM/lstm/lstm_cell/recurrent_kernelLSTM/lstm/lstm_cell/biastotalcount.Adagrad/LSTM/lstm/lstm_cell/kernel/accumulator8Adagrad/LSTM/lstm/lstm_cell/recurrent_kernel/accumulator,Adagrad/LSTM/lstm/lstm_cell/bias/accumulator*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__traced_restore_5290278��
�@
�
while_body_5289385
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
0while_lstm_cell_matmul_readvariableop_resource_06
2while_lstm_cell_matmul_1_readvariableop_resource_05
1while_lstm_cell_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
.while_lstm_cell_matmul_readvariableop_resource4
0while_lstm_cell_matmul_1_readvariableop_resource3
/while_lstm_cell_biasadd_readvariableop_resource��&while/lstm_cell/BiasAdd/ReadVariableOp�%while/lstm_cell/MatMul/ReadVariableOp�'while/lstm_cell/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02'
%while/lstm_cell/MatMul/ReadVariableOp�
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/MatMul�
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype02)
'while/lstm_cell/MatMul_1/ReadVariableOp�
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/MatMul_1�
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/add�
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02(
&while/lstm_cell/BiasAdd/ReadVariableOp�
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/BiasAddp
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const�
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim�
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
while/lstm_cell/split�
while/lstm_cell/TanhTanhwhile/lstm_cell/split:output:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/Tanh�
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/split:output:1*
T0*(
_output_shapes
:����������2
while/lstm_cell/Tanh_1�
while/lstm_cell/mulMulwhile/lstm_cell/Tanh_1:y:0while_placeholder_3*
T0*(
_output_shapes
:����������2
while/lstm_cell/mul�
while/lstm_cell/Tanh_2Tanhwhile/lstm_cell/split:output:2*
T0*(
_output_shapes
:����������2
while/lstm_cell/Tanh_2�
while/lstm_cell/mul_1Mulwhile/lstm_cell/Tanh:y:0while/lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/mul_1�
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/add_1�
while/lstm_cell/Tanh_3Tanhwhile/lstm_cell/split:output:3*
T0*(
_output_shapes
:����������2
while/lstm_cell/Tanh_3�
while/lstm_cell/Tanh_4Tanhwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/Tanh_4�
while/lstm_cell/mul_2Mulwhile/lstm_cell/Tanh_3:y:0while/lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell/add_1:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :����������:����������: : :::2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: 
�
�
&__inference_LSTM_layer_call_fn_5289294

inputs
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_LSTM_layer_call_and_return_conditional_losses_52881952
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������d:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
'__inference_model_layer_call_fn_5288546
input_1
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_52885372
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������d:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:���������d
!
_user_specified_name	input_1
�r
�
A__inference_lstm_layer_call_and_return_conditional_losses_5289999

inputs,
(lstm_cell_matmul_readvariableop_resource.
*lstm_cell_matmul_1_readvariableop_resource-
)lstm_cell_biasadd_readvariableop_resource
identity��<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp� lstm_cell/BiasAdd/ReadVariableOp�lstm_cell/MatMul/ReadVariableOp�!lstm_cell/MatMul_1/ReadVariableOp�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:����������2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:����������2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:d���������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02!
lstm_cell/MatMul/ReadVariableOp�
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell/MatMul�
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp�
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell/MatMul_1�
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell/add�
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp�
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell/BiasAddd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim�
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
lstm_cell/splitu
lstm_cell/TanhTanhlstm_cell/split:output:0*
T0*(
_output_shapes
:����������2
lstm_cell/Tanhy
lstm_cell/Tanh_1Tanhlstm_cell/split:output:1*
T0*(
_output_shapes
:����������2
lstm_cell/Tanh_1�
lstm_cell/mulMullstm_cell/Tanh_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:����������2
lstm_cell/muly
lstm_cell/Tanh_2Tanhlstm_cell/split:output:2*
T0*(
_output_shapes
:����������2
lstm_cell/Tanh_2�
lstm_cell/mul_1Mullstm_cell/Tanh:y:0lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:����������2
lstm_cell/mul_1�
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:����������2
lstm_cell/add_1y
lstm_cell/Tanh_3Tanhlstm_cell/split:output:3*
T0*(
_output_shapes
:����������2
lstm_cell/Tanh_3t
lstm_cell/Tanh_4Tanhlstm_cell/add_1:z:0*
T0*(
_output_shapes
:����������2
lstm_cell/Tanh_4�
lstm_cell/mul_2Mullstm_cell/Tanh_3:y:0lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:����������2
lstm_cell/mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :����������:����������: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_5289902*
condR
while_cond_5289901*M
output_shapes<
:: : : : :����������:����������: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:d����������*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:���������d�2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square�
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Const�
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum�
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x�
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mul�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp�
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul�
IdentityIdentitystrided_slice_3:output:0=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������d:::2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
while_cond_5289384
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_5289384___redundant_placeholder05
1while_while_cond_5289384___redundant_placeholder15
1while_while_cond_5289384___redundant_placeholder25
1while_while_cond_5289384___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :����������:����������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_5289901
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_5289901___redundant_placeholder05
1while_while_cond_5289901___redundant_placeholder15
1while_while_cond_5289901___redundant_placeholder25
1while_while_cond_5289901___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :����������:����������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
:
��
�
B__inference_model_layer_call_and_return_conditional_losses_5288919

inputs6
2lstm_lstm_lstm_cell_matmul_readvariableop_resource8
4lstm_lstm_lstm_cell_matmul_1_readvariableop_resource7
3lstm_lstm_lstm_cell_biasadd_readvariableop_resource
identity��*LSTM/lstm/lstm_cell/BiasAdd/ReadVariableOp�)LSTM/lstm/lstm_cell/MatMul/ReadVariableOp�+LSTM/lstm/lstm_cell/MatMul_1/ReadVariableOp�<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp�LSTM/lstm/whileX
LSTM/lstm/ShapeShapeinputs*
T0*
_output_shapes
:2
LSTM/lstm/Shape�
LSTM/lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
LSTM/lstm/strided_slice/stack�
LSTM/lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
LSTM/lstm/strided_slice/stack_1�
LSTM/lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
LSTM/lstm/strided_slice/stack_2�
LSTM/lstm/strided_sliceStridedSliceLSTM/lstm/Shape:output:0&LSTM/lstm/strided_slice/stack:output:0(LSTM/lstm/strided_slice/stack_1:output:0(LSTM/lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
LSTM/lstm/strided_sliceq
LSTM/lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
LSTM/lstm/zeros/mul/y�
LSTM/lstm/zeros/mulMul LSTM/lstm/strided_slice:output:0LSTM/lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
LSTM/lstm/zeros/muls
LSTM/lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
LSTM/lstm/zeros/Less/y�
LSTM/lstm/zeros/LessLessLSTM/lstm/zeros/mul:z:0LSTM/lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
LSTM/lstm/zeros/Lessw
LSTM/lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
LSTM/lstm/zeros/packed/1�
LSTM/lstm/zeros/packedPack LSTM/lstm/strided_slice:output:0!LSTM/lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
LSTM/lstm/zeros/packeds
LSTM/lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
LSTM/lstm/zeros/Const�
LSTM/lstm/zerosFillLSTM/lstm/zeros/packed:output:0LSTM/lstm/zeros/Const:output:0*
T0*(
_output_shapes
:����������2
LSTM/lstm/zerosu
LSTM/lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
LSTM/lstm/zeros_1/mul/y�
LSTM/lstm/zeros_1/mulMul LSTM/lstm/strided_slice:output:0 LSTM/lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
LSTM/lstm/zeros_1/mulw
LSTM/lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
LSTM/lstm/zeros_1/Less/y�
LSTM/lstm/zeros_1/LessLessLSTM/lstm/zeros_1/mul:z:0!LSTM/lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
LSTM/lstm/zeros_1/Less{
LSTM/lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
LSTM/lstm/zeros_1/packed/1�
LSTM/lstm/zeros_1/packedPack LSTM/lstm/strided_slice:output:0#LSTM/lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
LSTM/lstm/zeros_1/packedw
LSTM/lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
LSTM/lstm/zeros_1/Const�
LSTM/lstm/zeros_1Fill!LSTM/lstm/zeros_1/packed:output:0 LSTM/lstm/zeros_1/Const:output:0*
T0*(
_output_shapes
:����������2
LSTM/lstm/zeros_1�
LSTM/lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
LSTM/lstm/transpose/perm�
LSTM/lstm/transpose	Transposeinputs!LSTM/lstm/transpose/perm:output:0*
T0*+
_output_shapes
:d���������2
LSTM/lstm/transposem
LSTM/lstm/Shape_1ShapeLSTM/lstm/transpose:y:0*
T0*
_output_shapes
:2
LSTM/lstm/Shape_1�
LSTM/lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
LSTM/lstm/strided_slice_1/stack�
!LSTM/lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!LSTM/lstm/strided_slice_1/stack_1�
!LSTM/lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!LSTM/lstm/strided_slice_1/stack_2�
LSTM/lstm/strided_slice_1StridedSliceLSTM/lstm/Shape_1:output:0(LSTM/lstm/strided_slice_1/stack:output:0*LSTM/lstm/strided_slice_1/stack_1:output:0*LSTM/lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
LSTM/lstm/strided_slice_1�
%LSTM/lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%LSTM/lstm/TensorArrayV2/element_shape�
LSTM/lstm/TensorArrayV2TensorListReserve.LSTM/lstm/TensorArrayV2/element_shape:output:0"LSTM/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
LSTM/lstm/TensorArrayV2�
?LSTM/lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2A
?LSTM/lstm/TensorArrayUnstack/TensorListFromTensor/element_shape�
1LSTM/lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorLSTM/lstm/transpose:y:0HLSTM/lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type023
1LSTM/lstm/TensorArrayUnstack/TensorListFromTensor�
LSTM/lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
LSTM/lstm/strided_slice_2/stack�
!LSTM/lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!LSTM/lstm/strided_slice_2/stack_1�
!LSTM/lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!LSTM/lstm/strided_slice_2/stack_2�
LSTM/lstm/strided_slice_2StridedSliceLSTM/lstm/transpose:y:0(LSTM/lstm/strided_slice_2/stack:output:0*LSTM/lstm/strided_slice_2/stack_1:output:0*LSTM/lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
LSTM/lstm/strided_slice_2�
)LSTM/lstm/lstm_cell/MatMul/ReadVariableOpReadVariableOp2lstm_lstm_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02+
)LSTM/lstm/lstm_cell/MatMul/ReadVariableOp�
LSTM/lstm/lstm_cell/MatMulMatMul"LSTM/lstm/strided_slice_2:output:01LSTM/lstm/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
LSTM/lstm/lstm_cell/MatMul�
+LSTM/lstm/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp4lstm_lstm_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+LSTM/lstm/lstm_cell/MatMul_1/ReadVariableOp�
LSTM/lstm/lstm_cell/MatMul_1MatMulLSTM/lstm/zeros:output:03LSTM/lstm/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
LSTM/lstm/lstm_cell/MatMul_1�
LSTM/lstm/lstm_cell/addAddV2$LSTM/lstm/lstm_cell/MatMul:product:0&LSTM/lstm/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
LSTM/lstm/lstm_cell/add�
*LSTM/lstm/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp3lstm_lstm_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02,
*LSTM/lstm/lstm_cell/BiasAdd/ReadVariableOp�
LSTM/lstm/lstm_cell/BiasAddBiasAddLSTM/lstm/lstm_cell/add:z:02LSTM/lstm/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
LSTM/lstm/lstm_cell/BiasAddx
LSTM/lstm/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
LSTM/lstm/lstm_cell/Const�
#LSTM/lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#LSTM/lstm/lstm_cell/split/split_dim�
LSTM/lstm/lstm_cell/splitSplit,LSTM/lstm/lstm_cell/split/split_dim:output:0$LSTM/lstm/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
LSTM/lstm/lstm_cell/split�
LSTM/lstm/lstm_cell/TanhTanh"LSTM/lstm/lstm_cell/split:output:0*
T0*(
_output_shapes
:����������2
LSTM/lstm/lstm_cell/Tanh�
LSTM/lstm/lstm_cell/Tanh_1Tanh"LSTM/lstm/lstm_cell/split:output:1*
T0*(
_output_shapes
:����������2
LSTM/lstm/lstm_cell/Tanh_1�
LSTM/lstm/lstm_cell/mulMulLSTM/lstm/lstm_cell/Tanh_1:y:0LSTM/lstm/zeros_1:output:0*
T0*(
_output_shapes
:����������2
LSTM/lstm/lstm_cell/mul�
LSTM/lstm/lstm_cell/Tanh_2Tanh"LSTM/lstm/lstm_cell/split:output:2*
T0*(
_output_shapes
:����������2
LSTM/lstm/lstm_cell/Tanh_2�
LSTM/lstm/lstm_cell/mul_1MulLSTM/lstm/lstm_cell/Tanh:y:0LSTM/lstm/lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:����������2
LSTM/lstm/lstm_cell/mul_1�
LSTM/lstm/lstm_cell/add_1AddV2LSTM/lstm/lstm_cell/mul:z:0LSTM/lstm/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:����������2
LSTM/lstm/lstm_cell/add_1�
LSTM/lstm/lstm_cell/Tanh_3Tanh"LSTM/lstm/lstm_cell/split:output:3*
T0*(
_output_shapes
:����������2
LSTM/lstm/lstm_cell/Tanh_3�
LSTM/lstm/lstm_cell/Tanh_4TanhLSTM/lstm/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:����������2
LSTM/lstm/lstm_cell/Tanh_4�
LSTM/lstm/lstm_cell/mul_2MulLSTM/lstm/lstm_cell/Tanh_3:y:0LSTM/lstm/lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:����������2
LSTM/lstm/lstm_cell/mul_2�
'LSTM/lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   2)
'LSTM/lstm/TensorArrayV2_1/element_shape�
LSTM/lstm/TensorArrayV2_1TensorListReserve0LSTM/lstm/TensorArrayV2_1/element_shape:output:0"LSTM/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
LSTM/lstm/TensorArrayV2_1b
LSTM/lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
LSTM/lstm/time�
"LSTM/lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2$
"LSTM/lstm/while/maximum_iterations~
LSTM/lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
LSTM/lstm/while/loop_counter�
LSTM/lstm/whileWhile%LSTM/lstm/while/loop_counter:output:0+LSTM/lstm/while/maximum_iterations:output:0LSTM/lstm/time:output:0"LSTM/lstm/TensorArrayV2_1:handle:0LSTM/lstm/zeros:output:0LSTM/lstm/zeros_1:output:0"LSTM/lstm/strided_slice_1:output:0ALSTM/lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:02lstm_lstm_lstm_cell_matmul_readvariableop_resource4lstm_lstm_lstm_cell_matmul_1_readvariableop_resource3lstm_lstm_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :����������:����������: : : : : *%
_read_only_resource_inputs
	
*(
body R
LSTM_lstm_while_body_5288815*(
cond R
LSTM_lstm_while_cond_5288814*M
output_shapes<
:: : : : :����������:����������: : : : : *
parallel_iterations 2
LSTM/lstm/while�
:LSTM/lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   2<
:LSTM/lstm/TensorArrayV2Stack/TensorListStack/element_shape�
,LSTM/lstm/TensorArrayV2Stack/TensorListStackTensorListStackLSTM/lstm/while:output:3CLSTM/lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:d����������*
element_dtype02.
,LSTM/lstm/TensorArrayV2Stack/TensorListStack�
LSTM/lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2!
LSTM/lstm/strided_slice_3/stack�
!LSTM/lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!LSTM/lstm/strided_slice_3/stack_1�
!LSTM/lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!LSTM/lstm/strided_slice_3/stack_2�
LSTM/lstm/strided_slice_3StridedSlice5LSTM/lstm/TensorArrayV2Stack/TensorListStack:tensor:0(LSTM/lstm/strided_slice_3/stack:output:0*LSTM/lstm/strided_slice_3/stack_1:output:0*LSTM/lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask2
LSTM/lstm/strided_slice_3�
LSTM/lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
LSTM/lstm/transpose_1/perm�
LSTM/lstm/transpose_1	Transpose5LSTM/lstm/TensorArrayV2Stack/TensorListStack:tensor:0#LSTM/lstm/transpose_1/perm:output:0*
T0*,
_output_shapes
:���������d�2
LSTM/lstm/transpose_1z
LSTM/lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
LSTM/lstm/runtime�
(tf.math.l2_normalize/l2_normalize/SquareSquare"LSTM/lstm/strided_slice_3:output:0*
T0*(
_output_shapes
:����������2*
(tf.math.l2_normalize/l2_normalize/Square�
7tf.math.l2_normalize/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :29
7tf.math.l2_normalize/l2_normalize/Sum/reduction_indices�
%tf.math.l2_normalize/l2_normalize/SumSum,tf.math.l2_normalize/l2_normalize/Square:y:0@tf.math.l2_normalize/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(2'
%tf.math.l2_normalize/l2_normalize/Sum�
+tf.math.l2_normalize/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���.2-
+tf.math.l2_normalize/l2_normalize/Maximum/y�
)tf.math.l2_normalize/l2_normalize/MaximumMaximum.tf.math.l2_normalize/l2_normalize/Sum:output:04tf.math.l2_normalize/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:���������2+
)tf.math.l2_normalize/l2_normalize/Maximum�
'tf.math.l2_normalize/l2_normalize/RsqrtRsqrt-tf.math.l2_normalize/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:���������2)
'tf.math.l2_normalize/l2_normalize/Rsqrt�
!tf.math.l2_normalize/l2_normalizeMul"LSTM/lstm/strided_slice_3:output:0+tf.math.l2_normalize/l2_normalize/Rsqrt:y:0*
T0*(
_output_shapes
:����������2#
!tf.math.l2_normalize/l2_normalize�
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp2lstm_lstm_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square�
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Const�
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum�
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x�
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mul�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp4lstm_lstm_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp�
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul�
IdentityIdentity%tf.math.l2_normalize/l2_normalize:z:0+^LSTM/lstm/lstm_cell/BiasAdd/ReadVariableOp*^LSTM/lstm/lstm_cell/MatMul/ReadVariableOp,^LSTM/lstm/lstm_cell/MatMul_1/ReadVariableOp=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^LSTM/lstm/while*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������d:::2X
*LSTM/lstm/lstm_cell/BiasAdd/ReadVariableOp*LSTM/lstm/lstm_cell/BiasAdd/ReadVariableOp2V
)LSTM/lstm/lstm_cell/MatMul/ReadVariableOp)LSTM/lstm/lstm_cell/MatMul/ReadVariableOp2Z
+LSTM/lstm/lstm_cell/MatMul_1/ReadVariableOp+LSTM/lstm/lstm_cell/MatMul_1/ReadVariableOp2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2"
LSTM/lstm/whileLSTM/lstm/while:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�

�
LSTM_lstm_while_cond_52888140
,lstm_lstm_while_lstm_lstm_while_loop_counter6
2lstm_lstm_while_lstm_lstm_while_maximum_iterations
lstm_lstm_while_placeholder!
lstm_lstm_while_placeholder_1!
lstm_lstm_while_placeholder_2!
lstm_lstm_while_placeholder_32
.lstm_lstm_while_less_lstm_lstm_strided_slice_1I
Elstm_lstm_while_lstm_lstm_while_cond_5288814___redundant_placeholder0I
Elstm_lstm_while_lstm_lstm_while_cond_5288814___redundant_placeholder1I
Elstm_lstm_while_lstm_lstm_while_cond_5288814___redundant_placeholder2I
Elstm_lstm_while_lstm_lstm_while_cond_5288814___redundant_placeholder3
lstm_lstm_while_identity
�
LSTM/lstm/while/LessLesslstm_lstm_while_placeholder.lstm_lstm_while_less_lstm_lstm_strided_slice_1*
T0*
_output_shapes
: 2
LSTM/lstm/while/Less{
LSTM/lstm/while/IdentityIdentityLSTM/lstm/while/Less:z:0*
T0
*
_output_shapes
: 2
LSTM/lstm/while/Identity"=
lstm_lstm_while_identity!LSTM/lstm/while/Identity:output:0*U
_input_shapesD
B: : : : :����������:����������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
:
�3
�
#__inference__traced_restore_5290278
file_prefix!
assignvariableop_adagrad_iter$
 assignvariableop_1_adagrad_decay,
(assignvariableop_2_adagrad_learning_rate1
-assignvariableop_3_lstm_lstm_lstm_cell_kernel;
7assignvariableop_4_lstm_lstm_lstm_cell_recurrent_kernel/
+assignvariableop_5_lstm_lstm_lstm_cell_bias
assignvariableop_6_total
assignvariableop_7_countE
Aassignvariableop_8_adagrad_lstm_lstm_lstm_cell_kernel_accumulatorO
Kassignvariableop_9_adagrad_lstm_lstm_lstm_cell_recurrent_kernel_accumulatorD
@assignvariableop_10_adagrad_lstm_lstm_lstm_cell_bias_accumulator
identity_12��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBVtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBVtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBVtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*+
value"B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*D
_output_shapes2
0::::::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_adagrad_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp assignvariableop_1_adagrad_decayIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp(assignvariableop_2_adagrad_learning_rateIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp-assignvariableop_3_lstm_lstm_lstm_cell_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp7assignvariableop_4_lstm_lstm_lstm_cell_recurrent_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp+assignvariableop_5_lstm_lstm_lstm_cell_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpassignvariableop_6_totalIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_countIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpAassignvariableop_8_adagrad_lstm_lstm_lstm_cell_kernel_accumulatorIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpKassignvariableop_9_adagrad_lstm_lstm_lstm_cell_recurrent_kernel_accumulatorIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp@assignvariableop_10_adagrad_lstm_lstm_lstm_cell_bias_accumulatorIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_109
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_11Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_11�
Identity_12IdentityIdentity_11:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_12"#
identity_12Identity_12:output:0*A
_input_shapes0
.: :::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�G
�	
lstm_while_body_5289186&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3%
!lstm_while_lstm_strided_slice_1_0a
]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_09
5lstm_while_lstm_cell_matmul_readvariableop_resource_0;
7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0:
6lstm_while_lstm_cell_biasadd_readvariableop_resource_0
lstm_while_identity
lstm_while_identity_1
lstm_while_identity_2
lstm_while_identity_3
lstm_while_identity_4
lstm_while_identity_5#
lstm_while_lstm_strided_slice_1_
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor7
3lstm_while_lstm_cell_matmul_readvariableop_resource9
5lstm_while_lstm_cell_matmul_1_readvariableop_resource8
4lstm_while_lstm_cell_biasadd_readvariableop_resource��+lstm/while/lstm_cell/BiasAdd/ReadVariableOp�*lstm/while/lstm_cell/MatMul/ReadVariableOp�,lstm/while/lstm_cell/MatMul_1/ReadVariableOp�
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2>
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape�
.lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0lstm_while_placeholderElstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype020
.lstm/while/TensorArrayV2Read/TensorListGetItem�
*lstm/while/lstm_cell/MatMul/ReadVariableOpReadVariableOp5lstm_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02,
*lstm/while/lstm_cell/MatMul/ReadVariableOp�
lstm/while/lstm_cell/MatMulMatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:02lstm/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm/while/lstm_cell/MatMul�
,lstm/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype02.
,lstm/while/lstm_cell/MatMul_1/ReadVariableOp�
lstm/while/lstm_cell/MatMul_1MatMullstm_while_placeholder_24lstm/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm/while/lstm_cell/MatMul_1�
lstm/while/lstm_cell/addAddV2%lstm/while/lstm_cell/MatMul:product:0'lstm/while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm/while/lstm_cell/add�
+lstm/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp6lstm_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02-
+lstm/while/lstm_cell/BiasAdd/ReadVariableOp�
lstm/while/lstm_cell/BiasAddBiasAddlstm/while/lstm_cell/add:z:03lstm/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm/while/lstm_cell/BiasAddz
lstm/while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/lstm_cell/Const�
$lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm/while/lstm_cell/split/split_dim�
lstm/while/lstm_cell/splitSplit-lstm/while/lstm_cell/split/split_dim:output:0%lstm/while/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
lstm/while/lstm_cell/split�
lstm/while/lstm_cell/TanhTanh#lstm/while/lstm_cell/split:output:0*
T0*(
_output_shapes
:����������2
lstm/while/lstm_cell/Tanh�
lstm/while/lstm_cell/Tanh_1Tanh#lstm/while/lstm_cell/split:output:1*
T0*(
_output_shapes
:����������2
lstm/while/lstm_cell/Tanh_1�
lstm/while/lstm_cell/mulMullstm/while/lstm_cell/Tanh_1:y:0lstm_while_placeholder_3*
T0*(
_output_shapes
:����������2
lstm/while/lstm_cell/mul�
lstm/while/lstm_cell/Tanh_2Tanh#lstm/while/lstm_cell/split:output:2*
T0*(
_output_shapes
:����������2
lstm/while/lstm_cell/Tanh_2�
lstm/while/lstm_cell/mul_1Mullstm/while/lstm_cell/Tanh:y:0lstm/while/lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:����������2
lstm/while/lstm_cell/mul_1�
lstm/while/lstm_cell/add_1AddV2lstm/while/lstm_cell/mul:z:0lstm/while/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:����������2
lstm/while/lstm_cell/add_1�
lstm/while/lstm_cell/Tanh_3Tanh#lstm/while/lstm_cell/split:output:3*
T0*(
_output_shapes
:����������2
lstm/while/lstm_cell/Tanh_3�
lstm/while/lstm_cell/Tanh_4Tanhlstm/while/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:����������2
lstm/while/lstm_cell/Tanh_4�
lstm/while/lstm_cell/mul_2Mullstm/while/lstm_cell/Tanh_3:y:0lstm/while/lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:����������2
lstm/while/lstm_cell/mul_2�
/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_while_placeholder_1lstm_while_placeholderlstm/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype021
/lstm/while/TensorArrayV2Write/TensorListSetItemf
lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/add/y}
lstm/while/addAddV2lstm_while_placeholderlstm/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm/while/addj
lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/add_1/y�
lstm/while/add_1AddV2"lstm_while_lstm_while_loop_counterlstm/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm/while/add_1�
lstm/while/IdentityIdentitylstm/while/add_1:z:0,^lstm/while/lstm_cell/BiasAdd/ReadVariableOp+^lstm/while/lstm_cell/MatMul/ReadVariableOp-^lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm/while/Identity�
lstm/while/Identity_1Identity(lstm_while_lstm_while_maximum_iterations,^lstm/while/lstm_cell/BiasAdd/ReadVariableOp+^lstm/while/lstm_cell/MatMul/ReadVariableOp-^lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm/while/Identity_1�
lstm/while/Identity_2Identitylstm/while/add:z:0,^lstm/while/lstm_cell/BiasAdd/ReadVariableOp+^lstm/while/lstm_cell/MatMul/ReadVariableOp-^lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm/while/Identity_2�
lstm/while/Identity_3Identity?lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0,^lstm/while/lstm_cell/BiasAdd/ReadVariableOp+^lstm/while/lstm_cell/MatMul/ReadVariableOp-^lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm/while/Identity_3�
lstm/while/Identity_4Identitylstm/while/lstm_cell/mul_2:z:0,^lstm/while/lstm_cell/BiasAdd/ReadVariableOp+^lstm/while/lstm_cell/MatMul/ReadVariableOp-^lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2
lstm/while/Identity_4�
lstm/while/Identity_5Identitylstm/while/lstm_cell/add_1:z:0,^lstm/while/lstm_cell/BiasAdd/ReadVariableOp+^lstm/while/lstm_cell/MatMul/ReadVariableOp-^lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2
lstm/while/Identity_5"3
lstm_while_identitylstm/while/Identity:output:0"7
lstm_while_identity_1lstm/while/Identity_1:output:0"7
lstm_while_identity_2lstm/while/Identity_2:output:0"7
lstm_while_identity_3lstm/while/Identity_3:output:0"7
lstm_while_identity_4lstm/while/Identity_4:output:0"7
lstm_while_identity_5lstm/while/Identity_5:output:0"n
4lstm_while_lstm_cell_biasadd_readvariableop_resource6lstm_while_lstm_cell_biasadd_readvariableop_resource_0"p
5lstm_while_lstm_cell_matmul_1_readvariableop_resource7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0"l
3lstm_while_lstm_cell_matmul_readvariableop_resource5lstm_while_lstm_cell_matmul_readvariableop_resource_0"D
lstm_while_lstm_strided_slice_1!lstm_while_lstm_strided_slice_1_0"�
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :����������:����������: : :::2Z
+lstm/while/lstm_cell/BiasAdd/ReadVariableOp+lstm/while/lstm_cell/BiasAdd/ReadVariableOp2X
*lstm/while/lstm_cell/MatMul/ReadVariableOp*lstm/while/lstm_cell/MatMul/ReadVariableOp2\
,lstm/while/lstm_cell/MatMul_1/ReadVariableOp,lstm/while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: 
�@
�
while_body_5289902
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
0while_lstm_cell_matmul_readvariableop_resource_06
2while_lstm_cell_matmul_1_readvariableop_resource_05
1while_lstm_cell_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
.while_lstm_cell_matmul_readvariableop_resource4
0while_lstm_cell_matmul_1_readvariableop_resource3
/while_lstm_cell_biasadd_readvariableop_resource��&while/lstm_cell/BiasAdd/ReadVariableOp�%while/lstm_cell/MatMul/ReadVariableOp�'while/lstm_cell/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02'
%while/lstm_cell/MatMul/ReadVariableOp�
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/MatMul�
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype02)
'while/lstm_cell/MatMul_1/ReadVariableOp�
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/MatMul_1�
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/add�
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02(
&while/lstm_cell/BiasAdd/ReadVariableOp�
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/BiasAddp
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const�
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim�
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
while/lstm_cell/split�
while/lstm_cell/TanhTanhwhile/lstm_cell/split:output:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/Tanh�
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/split:output:1*
T0*(
_output_shapes
:����������2
while/lstm_cell/Tanh_1�
while/lstm_cell/mulMulwhile/lstm_cell/Tanh_1:y:0while_placeholder_3*
T0*(
_output_shapes
:����������2
while/lstm_cell/mul�
while/lstm_cell/Tanh_2Tanhwhile/lstm_cell/split:output:2*
T0*(
_output_shapes
:����������2
while/lstm_cell/Tanh_2�
while/lstm_cell/mul_1Mulwhile/lstm_cell/Tanh:y:0while/lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/mul_1�
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/add_1�
while/lstm_cell/Tanh_3Tanhwhile/lstm_cell/split:output:3*
T0*(
_output_shapes
:����������2
while/lstm_cell/Tanh_3�
while/lstm_cell/Tanh_4Tanhwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/Tanh_4�
while/lstm_cell/mul_2Mulwhile/lstm_cell/Tanh_3:y:0while/lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell/add_1:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :����������:����������: : :::2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: 
�O
�

LSTM_lstm_while_body_52888150
,lstm_lstm_while_lstm_lstm_while_loop_counter6
2lstm_lstm_while_lstm_lstm_while_maximum_iterations
lstm_lstm_while_placeholder!
lstm_lstm_while_placeholder_1!
lstm_lstm_while_placeholder_2!
lstm_lstm_while_placeholder_3/
+lstm_lstm_while_lstm_lstm_strided_slice_1_0k
glstm_lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_lstm_tensorarrayunstack_tensorlistfromtensor_0>
:lstm_lstm_while_lstm_cell_matmul_readvariableop_resource_0@
<lstm_lstm_while_lstm_cell_matmul_1_readvariableop_resource_0?
;lstm_lstm_while_lstm_cell_biasadd_readvariableop_resource_0
lstm_lstm_while_identity
lstm_lstm_while_identity_1
lstm_lstm_while_identity_2
lstm_lstm_while_identity_3
lstm_lstm_while_identity_4
lstm_lstm_while_identity_5-
)lstm_lstm_while_lstm_lstm_strided_slice_1i
elstm_lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_lstm_tensorarrayunstack_tensorlistfromtensor<
8lstm_lstm_while_lstm_cell_matmul_readvariableop_resource>
:lstm_lstm_while_lstm_cell_matmul_1_readvariableop_resource=
9lstm_lstm_while_lstm_cell_biasadd_readvariableop_resource��0LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp�/LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp�1LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp�
ALSTM/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2C
ALSTM/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape�
3LSTM/lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemglstm_lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_lstm_tensorarrayunstack_tensorlistfromtensor_0lstm_lstm_while_placeholderJLSTM/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype025
3LSTM/lstm/while/TensorArrayV2Read/TensorListGetItem�
/LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOpReadVariableOp:lstm_lstm_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype021
/LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp�
 LSTM/lstm/while/lstm_cell/MatMulMatMul:LSTM/lstm/while/TensorArrayV2Read/TensorListGetItem:item:07LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2"
 LSTM/lstm/while/lstm_cell/MatMul�
1LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp<lstm_lstm_while_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype023
1LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp�
"LSTM/lstm/while/lstm_cell/MatMul_1MatMullstm_lstm_while_placeholder_29LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2$
"LSTM/lstm/while/lstm_cell/MatMul_1�
LSTM/lstm/while/lstm_cell/addAddV2*LSTM/lstm/while/lstm_cell/MatMul:product:0,LSTM/lstm/while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
LSTM/lstm/while/lstm_cell/add�
0LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp;lstm_lstm_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype022
0LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp�
!LSTM/lstm/while/lstm_cell/BiasAddBiasAdd!LSTM/lstm/while/lstm_cell/add:z:08LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2#
!LSTM/lstm/while/lstm_cell/BiasAdd�
LSTM/lstm/while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2!
LSTM/lstm/while/lstm_cell/Const�
)LSTM/lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)LSTM/lstm/while/lstm_cell/split/split_dim�
LSTM/lstm/while/lstm_cell/splitSplit2LSTM/lstm/while/lstm_cell/split/split_dim:output:0*LSTM/lstm/while/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2!
LSTM/lstm/while/lstm_cell/split�
LSTM/lstm/while/lstm_cell/TanhTanh(LSTM/lstm/while/lstm_cell/split:output:0*
T0*(
_output_shapes
:����������2 
LSTM/lstm/while/lstm_cell/Tanh�
 LSTM/lstm/while/lstm_cell/Tanh_1Tanh(LSTM/lstm/while/lstm_cell/split:output:1*
T0*(
_output_shapes
:����������2"
 LSTM/lstm/while/lstm_cell/Tanh_1�
LSTM/lstm/while/lstm_cell/mulMul$LSTM/lstm/while/lstm_cell/Tanh_1:y:0lstm_lstm_while_placeholder_3*
T0*(
_output_shapes
:����������2
LSTM/lstm/while/lstm_cell/mul�
 LSTM/lstm/while/lstm_cell/Tanh_2Tanh(LSTM/lstm/while/lstm_cell/split:output:2*
T0*(
_output_shapes
:����������2"
 LSTM/lstm/while/lstm_cell/Tanh_2�
LSTM/lstm/while/lstm_cell/mul_1Mul"LSTM/lstm/while/lstm_cell/Tanh:y:0$LSTM/lstm/while/lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:����������2!
LSTM/lstm/while/lstm_cell/mul_1�
LSTM/lstm/while/lstm_cell/add_1AddV2!LSTM/lstm/while/lstm_cell/mul:z:0#LSTM/lstm/while/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:����������2!
LSTM/lstm/while/lstm_cell/add_1�
 LSTM/lstm/while/lstm_cell/Tanh_3Tanh(LSTM/lstm/while/lstm_cell/split:output:3*
T0*(
_output_shapes
:����������2"
 LSTM/lstm/while/lstm_cell/Tanh_3�
 LSTM/lstm/while/lstm_cell/Tanh_4Tanh#LSTM/lstm/while/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:����������2"
 LSTM/lstm/while/lstm_cell/Tanh_4�
LSTM/lstm/while/lstm_cell/mul_2Mul$LSTM/lstm/while/lstm_cell/Tanh_3:y:0$LSTM/lstm/while/lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:����������2!
LSTM/lstm/while/lstm_cell/mul_2�
4LSTM/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_lstm_while_placeholder_1lstm_lstm_while_placeholder#LSTM/lstm/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype026
4LSTM/lstm/while/TensorArrayV2Write/TensorListSetItemp
LSTM/lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
LSTM/lstm/while/add/y�
LSTM/lstm/while/addAddV2lstm_lstm_while_placeholderLSTM/lstm/while/add/y:output:0*
T0*
_output_shapes
: 2
LSTM/lstm/while/addt
LSTM/lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
LSTM/lstm/while/add_1/y�
LSTM/lstm/while/add_1AddV2,lstm_lstm_while_lstm_lstm_while_loop_counter LSTM/lstm/while/add_1/y:output:0*
T0*
_output_shapes
: 2
LSTM/lstm/while/add_1�
LSTM/lstm/while/IdentityIdentityLSTM/lstm/while/add_1:z:01^LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp0^LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp2^LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
LSTM/lstm/while/Identity�
LSTM/lstm/while/Identity_1Identity2lstm_lstm_while_lstm_lstm_while_maximum_iterations1^LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp0^LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp2^LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
LSTM/lstm/while/Identity_1�
LSTM/lstm/while/Identity_2IdentityLSTM/lstm/while/add:z:01^LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp0^LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp2^LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
LSTM/lstm/while/Identity_2�
LSTM/lstm/while/Identity_3IdentityDLSTM/lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:01^LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp0^LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp2^LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
LSTM/lstm/while/Identity_3�
LSTM/lstm/while/Identity_4Identity#LSTM/lstm/while/lstm_cell/mul_2:z:01^LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp0^LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp2^LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2
LSTM/lstm/while/Identity_4�
LSTM/lstm/while/Identity_5Identity#LSTM/lstm/while/lstm_cell/add_1:z:01^LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp0^LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp2^LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2
LSTM/lstm/while/Identity_5"=
lstm_lstm_while_identity!LSTM/lstm/while/Identity:output:0"A
lstm_lstm_while_identity_1#LSTM/lstm/while/Identity_1:output:0"A
lstm_lstm_while_identity_2#LSTM/lstm/while/Identity_2:output:0"A
lstm_lstm_while_identity_3#LSTM/lstm/while/Identity_3:output:0"A
lstm_lstm_while_identity_4#LSTM/lstm/while/Identity_4:output:0"A
lstm_lstm_while_identity_5#LSTM/lstm/while/Identity_5:output:0"x
9lstm_lstm_while_lstm_cell_biasadd_readvariableop_resource;lstm_lstm_while_lstm_cell_biasadd_readvariableop_resource_0"z
:lstm_lstm_while_lstm_cell_matmul_1_readvariableop_resource<lstm_lstm_while_lstm_cell_matmul_1_readvariableop_resource_0"v
8lstm_lstm_while_lstm_cell_matmul_readvariableop_resource:lstm_lstm_while_lstm_cell_matmul_readvariableop_resource_0"X
)lstm_lstm_while_lstm_lstm_strided_slice_1+lstm_lstm_while_lstm_lstm_strided_slice_1_0"�
elstm_lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_lstm_tensorarrayunstack_tensorlistfromtensorglstm_lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_lstm_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :����������:����������: : :::2d
0LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp0LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp2b
/LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp/LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp2f
1LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp1LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: 
�,
�
B__inference_model_layer_call_and_return_conditional_losses_5288463
input_1
lstm_5288436
lstm_5288438
lstm_5288440
identity��LSTM/StatefulPartitionedCall�<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp�
LSTM/StatefulPartitionedCallStatefulPartitionedCallinput_1lstm_5288436lstm_5288438lstm_5288440*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_LSTM_layer_call_and_return_conditional_losses_52881952
LSTM/StatefulPartitionedCall�
(tf.math.l2_normalize/l2_normalize/SquareSquare%LSTM/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:����������2*
(tf.math.l2_normalize/l2_normalize/Square�
7tf.math.l2_normalize/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :29
7tf.math.l2_normalize/l2_normalize/Sum/reduction_indices�
%tf.math.l2_normalize/l2_normalize/SumSum,tf.math.l2_normalize/l2_normalize/Square:y:0@tf.math.l2_normalize/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(2'
%tf.math.l2_normalize/l2_normalize/Sum�
+tf.math.l2_normalize/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���.2-
+tf.math.l2_normalize/l2_normalize/Maximum/y�
)tf.math.l2_normalize/l2_normalize/MaximumMaximum.tf.math.l2_normalize/l2_normalize/Sum:output:04tf.math.l2_normalize/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:���������2+
)tf.math.l2_normalize/l2_normalize/Maximum�
'tf.math.l2_normalize/l2_normalize/RsqrtRsqrt-tf.math.l2_normalize/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:���������2)
'tf.math.l2_normalize/l2_normalize/Rsqrt�
!tf.math.l2_normalize/l2_normalizeMul%LSTM/StatefulPartitionedCall:output:0+tf.math.l2_normalize/l2_normalize/Rsqrt:y:0*
T0*(
_output_shapes
:����������2#
!tf.math.l2_normalize/l2_normalize�
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_5288436*
_output_shapes
:	�*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square�
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Const�
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum�
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x�
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mul�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_5288438* 
_output_shapes
:
��*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp�
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul�
IdentityIdentity%tf.math.l2_normalize/l2_normalize:z:0^LSTM/StatefulPartitionedCall=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������d:::2<
LSTM/StatefulPartitionedCallLSTM/StatefulPartitionedCall2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:T P
+
_output_shapes
:���������d
!
_user_specified_name	input_1
�|
�
A__inference_LSTM_layer_call_and_return_conditional_losses_5288383

inputs1
-lstm_lstm_cell_matmul_readvariableop_resource3
/lstm_lstm_cell_matmul_1_readvariableop_resource2
.lstm_lstm_cell_biasadd_readvariableop_resource
identity��<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp�%lstm/lstm_cell/BiasAdd/ReadVariableOp�$lstm/lstm_cell/MatMul/ReadVariableOp�&lstm/lstm_cell/MatMul_1/ReadVariableOp�
lstm/whileN

lstm/ShapeShapeinputs*
T0*
_output_shapes
:2

lstm/Shape~
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice/stack�
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_1�
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_2�
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_sliceg
lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm/zeros/mul/y�
lstm/zeros/mulMullstm/strided_slice:output:0lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/muli
lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm/zeros/Less/y{
lstm/zeros/LessLesslstm/zeros/mul:z:0lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/Lessm
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
lstm/zeros/packed/1�
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros/packedi
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros/Const�

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*(
_output_shapes
:����������2

lstm/zerosk
lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm/zeros_1/mul/y�
lstm/zeros_1/mulMullstm/strided_slice:output:0lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/mulm
lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm/zeros_1/Less/y�
lstm/zeros_1/LessLesslstm/zeros_1/mul:z:0lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/Lessq
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
lstm/zeros_1/packed/1�
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros_1/packedm
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros_1/Const�
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*(
_output_shapes
:����������2
lstm/zeros_1
lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose/perm�
lstm/transpose	Transposeinputslstm/transpose/perm:output:0*
T0*+
_output_shapes
:d���������2
lstm/transpose^
lstm/Shape_1Shapelstm/transpose:y:0*
T0*
_output_shapes
:2
lstm/Shape_1�
lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_1/stack�
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_1�
lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_2�
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slice_1�
 lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 lstm/TensorArrayV2/element_shape�
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2�
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2<
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shape�
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02.
,lstm/TensorArrayUnstack/TensorListFromTensor�
lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_2/stack�
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_1�
lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_2�
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
lstm/strided_slice_2�
$lstm/lstm_cell/MatMul/ReadVariableOpReadVariableOp-lstm_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02&
$lstm/lstm_cell/MatMul/ReadVariableOp�
lstm/lstm_cell/MatMulMatMullstm/strided_slice_2:output:0,lstm/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm/lstm_cell/MatMul�
&lstm/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp/lstm_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype02(
&lstm/lstm_cell/MatMul_1/ReadVariableOp�
lstm/lstm_cell/MatMul_1MatMullstm/zeros:output:0.lstm/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm/lstm_cell/MatMul_1�
lstm/lstm_cell/addAddV2lstm/lstm_cell/MatMul:product:0!lstm/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm/lstm_cell/add�
%lstm/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp.lstm_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02'
%lstm/lstm_cell/BiasAdd/ReadVariableOp�
lstm/lstm_cell/BiasAddBiasAddlstm/lstm_cell/add:z:0-lstm/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm/lstm_cell/BiasAddn
lstm/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/lstm_cell/Const�
lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm/lstm_cell/split/split_dim�
lstm/lstm_cell/splitSplit'lstm/lstm_cell/split/split_dim:output:0lstm/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
lstm/lstm_cell/split�
lstm/lstm_cell/TanhTanhlstm/lstm_cell/split:output:0*
T0*(
_output_shapes
:����������2
lstm/lstm_cell/Tanh�
lstm/lstm_cell/Tanh_1Tanhlstm/lstm_cell/split:output:1*
T0*(
_output_shapes
:����������2
lstm/lstm_cell/Tanh_1�
lstm/lstm_cell/mulMullstm/lstm_cell/Tanh_1:y:0lstm/zeros_1:output:0*
T0*(
_output_shapes
:����������2
lstm/lstm_cell/mul�
lstm/lstm_cell/Tanh_2Tanhlstm/lstm_cell/split:output:2*
T0*(
_output_shapes
:����������2
lstm/lstm_cell/Tanh_2�
lstm/lstm_cell/mul_1Mullstm/lstm_cell/Tanh:y:0lstm/lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:����������2
lstm/lstm_cell/mul_1�
lstm/lstm_cell/add_1AddV2lstm/lstm_cell/mul:z:0lstm/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:����������2
lstm/lstm_cell/add_1�
lstm/lstm_cell/Tanh_3Tanhlstm/lstm_cell/split:output:3*
T0*(
_output_shapes
:����������2
lstm/lstm_cell/Tanh_3�
lstm/lstm_cell/Tanh_4Tanhlstm/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:����������2
lstm/lstm_cell/Tanh_4�
lstm/lstm_cell/mul_2Mullstm/lstm_cell/Tanh_3:y:0lstm/lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:����������2
lstm/lstm_cell/mul_2�
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   2$
"lstm/TensorArrayV2_1/element_shape�
lstm/TensorArrayV2_1TensorListReserve+lstm/TensorArrayV2_1/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2_1X
	lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
	lstm/time�
lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
lstm/while/maximum_iterationst
lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm/while/loop_counter�

lstm/whileWhile lstm/while/loop_counter:output:0&lstm/while/maximum_iterations:output:0lstm/time:output:0lstm/TensorArrayV2_1:handle:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/strided_slice_1:output:0<lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_lstm_cell_matmul_readvariableop_resource/lstm_lstm_cell_matmul_1_readvariableop_resource.lstm_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :����������:����������: : : : : *%
_read_only_resource_inputs
	
*#
bodyR
lstm_while_body_5288286*#
condR
lstm_while_cond_5288285*M
output_shapes<
:: : : : :����������:����������: : : : : *
parallel_iterations 2

lstm/while�
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   27
5lstm/TensorArrayV2Stack/TensorListStack/element_shape�
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:d����������*
element_dtype02)
'lstm/TensorArrayV2Stack/TensorListStack�
lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
lstm/strided_slice_3/stack�
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_3/stack_1�
lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_3/stack_2�
lstm/strided_slice_3StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask2
lstm/strided_slice_3�
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose_1/perm�
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*,
_output_shapes
:���������d�2
lstm/transpose_1p
lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/runtime�
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp-lstm_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square�
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Const�
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum�
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x�
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mul�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp/lstm_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp�
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul�
IdentityIdentitylstm/strided_slice_3:output:0=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp&^lstm/lstm_cell/BiasAdd/ReadVariableOp%^lstm/lstm_cell/MatMul/ReadVariableOp'^lstm/lstm_cell/MatMul_1/ReadVariableOp^lstm/while*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������d:::2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2N
%lstm/lstm_cell/BiasAdd/ReadVariableOp%lstm/lstm_cell/BiasAdd/ReadVariableOp2L
$lstm/lstm_cell/MatMul/ReadVariableOp$lstm/lstm_cell/MatMul/ReadVariableOp2P
&lstm/lstm_cell/MatMul_1/ReadVariableOp&lstm/lstm_cell/MatMul_1/ReadVariableOp2

lstm/while
lstm/while:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
'__inference_model_layer_call_fn_5288505
input_1
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_52884962
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������d:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:���������d
!
_user_specified_name	input_1
�
�
while_cond_5287535
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_5287535___redundant_placeholder05
1while_while_cond_5287535___redundant_placeholder15
1while_while_cond_5287535___redundant_placeholder25
1while_while_cond_5287535___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :����������:����������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
:
�
�
&__inference_LSTM_layer_call_fn_5288215
input_1
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_LSTM_layer_call_and_return_conditional_losses_52881952
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������d:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:���������d
!
_user_specified_name	input_1
�
�
__inference_loss_fn_1_5290179S
Olstm_lstm_lstm_cell_recurrent_kernel_regularizer_square_readvariableop_resource
identity��FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOpOlstm_lstm_lstm_cell_recurrent_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
��*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp�
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul�
IdentityIdentity8LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul:z:0G^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp
��
�
"__inference__wrapped_model_5287088
input_1<
8model_lstm_lstm_lstm_cell_matmul_readvariableop_resource>
:model_lstm_lstm_lstm_cell_matmul_1_readvariableop_resource=
9model_lstm_lstm_lstm_cell_biasadd_readvariableop_resource
identity��0model/LSTM/lstm/lstm_cell/BiasAdd/ReadVariableOp�/model/LSTM/lstm/lstm_cell/MatMul/ReadVariableOp�1model/LSTM/lstm/lstm_cell/MatMul_1/ReadVariableOp�model/LSTM/lstm/whilee
model/LSTM/lstm/ShapeShapeinput_1*
T0*
_output_shapes
:2
model/LSTM/lstm/Shape�
#model/LSTM/lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model/LSTM/lstm/strided_slice/stack�
%model/LSTM/lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%model/LSTM/lstm/strided_slice/stack_1�
%model/LSTM/lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%model/LSTM/lstm/strided_slice/stack_2�
model/LSTM/lstm/strided_sliceStridedSlicemodel/LSTM/lstm/Shape:output:0,model/LSTM/lstm/strided_slice/stack:output:0.model/LSTM/lstm/strided_slice/stack_1:output:0.model/LSTM/lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model/LSTM/lstm/strided_slice}
model/LSTM/lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
model/LSTM/lstm/zeros/mul/y�
model/LSTM/lstm/zeros/mulMul&model/LSTM/lstm/strided_slice:output:0$model/LSTM/lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
model/LSTM/lstm/zeros/mul
model/LSTM/lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
model/LSTM/lstm/zeros/Less/y�
model/LSTM/lstm/zeros/LessLessmodel/LSTM/lstm/zeros/mul:z:0%model/LSTM/lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
model/LSTM/lstm/zeros/Less�
model/LSTM/lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2 
model/LSTM/lstm/zeros/packed/1�
model/LSTM/lstm/zeros/packedPack&model/LSTM/lstm/strided_slice:output:0'model/LSTM/lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
model/LSTM/lstm/zeros/packed
model/LSTM/lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model/LSTM/lstm/zeros/Const�
model/LSTM/lstm/zerosFill%model/LSTM/lstm/zeros/packed:output:0$model/LSTM/lstm/zeros/Const:output:0*
T0*(
_output_shapes
:����������2
model/LSTM/lstm/zeros�
model/LSTM/lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
model/LSTM/lstm/zeros_1/mul/y�
model/LSTM/lstm/zeros_1/mulMul&model/LSTM/lstm/strided_slice:output:0&model/LSTM/lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
model/LSTM/lstm/zeros_1/mul�
model/LSTM/lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2 
model/LSTM/lstm/zeros_1/Less/y�
model/LSTM/lstm/zeros_1/LessLessmodel/LSTM/lstm/zeros_1/mul:z:0'model/LSTM/lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
model/LSTM/lstm/zeros_1/Less�
 model/LSTM/lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2"
 model/LSTM/lstm/zeros_1/packed/1�
model/LSTM/lstm/zeros_1/packedPack&model/LSTM/lstm/strided_slice:output:0)model/LSTM/lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
model/LSTM/lstm/zeros_1/packed�
model/LSTM/lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model/LSTM/lstm/zeros_1/Const�
model/LSTM/lstm/zeros_1Fill'model/LSTM/lstm/zeros_1/packed:output:0&model/LSTM/lstm/zeros_1/Const:output:0*
T0*(
_output_shapes
:����������2
model/LSTM/lstm/zeros_1�
model/LSTM/lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
model/LSTM/lstm/transpose/perm�
model/LSTM/lstm/transpose	Transposeinput_1'model/LSTM/lstm/transpose/perm:output:0*
T0*+
_output_shapes
:d���������2
model/LSTM/lstm/transpose
model/LSTM/lstm/Shape_1Shapemodel/LSTM/lstm/transpose:y:0*
T0*
_output_shapes
:2
model/LSTM/lstm/Shape_1�
%model/LSTM/lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%model/LSTM/lstm/strided_slice_1/stack�
'model/LSTM/lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'model/LSTM/lstm/strided_slice_1/stack_1�
'model/LSTM/lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'model/LSTM/lstm/strided_slice_1/stack_2�
model/LSTM/lstm/strided_slice_1StridedSlice model/LSTM/lstm/Shape_1:output:0.model/LSTM/lstm/strided_slice_1/stack:output:00model/LSTM/lstm/strided_slice_1/stack_1:output:00model/LSTM/lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
model/LSTM/lstm/strided_slice_1�
+model/LSTM/lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2-
+model/LSTM/lstm/TensorArrayV2/element_shape�
model/LSTM/lstm/TensorArrayV2TensorListReserve4model/LSTM/lstm/TensorArrayV2/element_shape:output:0(model/LSTM/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
model/LSTM/lstm/TensorArrayV2�
Emodel/LSTM/lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2G
Emodel/LSTM/lstm/TensorArrayUnstack/TensorListFromTensor/element_shape�
7model/LSTM/lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel/LSTM/lstm/transpose:y:0Nmodel/LSTM/lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7model/LSTM/lstm/TensorArrayUnstack/TensorListFromTensor�
%model/LSTM/lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%model/LSTM/lstm/strided_slice_2/stack�
'model/LSTM/lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'model/LSTM/lstm/strided_slice_2/stack_1�
'model/LSTM/lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'model/LSTM/lstm/strided_slice_2/stack_2�
model/LSTM/lstm/strided_slice_2StridedSlicemodel/LSTM/lstm/transpose:y:0.model/LSTM/lstm/strided_slice_2/stack:output:00model/LSTM/lstm/strided_slice_2/stack_1:output:00model/LSTM/lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2!
model/LSTM/lstm/strided_slice_2�
/model/LSTM/lstm/lstm_cell/MatMul/ReadVariableOpReadVariableOp8model_lstm_lstm_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype021
/model/LSTM/lstm/lstm_cell/MatMul/ReadVariableOp�
 model/LSTM/lstm/lstm_cell/MatMulMatMul(model/LSTM/lstm/strided_slice_2:output:07model/LSTM/lstm/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2"
 model/LSTM/lstm/lstm_cell/MatMul�
1model/LSTM/lstm/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp:model_lstm_lstm_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype023
1model/LSTM/lstm/lstm_cell/MatMul_1/ReadVariableOp�
"model/LSTM/lstm/lstm_cell/MatMul_1MatMulmodel/LSTM/lstm/zeros:output:09model/LSTM/lstm/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2$
"model/LSTM/lstm/lstm_cell/MatMul_1�
model/LSTM/lstm/lstm_cell/addAddV2*model/LSTM/lstm/lstm_cell/MatMul:product:0,model/LSTM/lstm/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
model/LSTM/lstm/lstm_cell/add�
0model/LSTM/lstm/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp9model_lstm_lstm_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype022
0model/LSTM/lstm/lstm_cell/BiasAdd/ReadVariableOp�
!model/LSTM/lstm/lstm_cell/BiasAddBiasAdd!model/LSTM/lstm/lstm_cell/add:z:08model/LSTM/lstm/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2#
!model/LSTM/lstm/lstm_cell/BiasAdd�
model/LSTM/lstm/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2!
model/LSTM/lstm/lstm_cell/Const�
)model/LSTM/lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)model/LSTM/lstm/lstm_cell/split/split_dim�
model/LSTM/lstm/lstm_cell/splitSplit2model/LSTM/lstm/lstm_cell/split/split_dim:output:0*model/LSTM/lstm/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2!
model/LSTM/lstm/lstm_cell/split�
model/LSTM/lstm/lstm_cell/TanhTanh(model/LSTM/lstm/lstm_cell/split:output:0*
T0*(
_output_shapes
:����������2 
model/LSTM/lstm/lstm_cell/Tanh�
 model/LSTM/lstm/lstm_cell/Tanh_1Tanh(model/LSTM/lstm/lstm_cell/split:output:1*
T0*(
_output_shapes
:����������2"
 model/LSTM/lstm/lstm_cell/Tanh_1�
model/LSTM/lstm/lstm_cell/mulMul$model/LSTM/lstm/lstm_cell/Tanh_1:y:0 model/LSTM/lstm/zeros_1:output:0*
T0*(
_output_shapes
:����������2
model/LSTM/lstm/lstm_cell/mul�
 model/LSTM/lstm/lstm_cell/Tanh_2Tanh(model/LSTM/lstm/lstm_cell/split:output:2*
T0*(
_output_shapes
:����������2"
 model/LSTM/lstm/lstm_cell/Tanh_2�
model/LSTM/lstm/lstm_cell/mul_1Mul"model/LSTM/lstm/lstm_cell/Tanh:y:0$model/LSTM/lstm/lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:����������2!
model/LSTM/lstm/lstm_cell/mul_1�
model/LSTM/lstm/lstm_cell/add_1AddV2!model/LSTM/lstm/lstm_cell/mul:z:0#model/LSTM/lstm/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:����������2!
model/LSTM/lstm/lstm_cell/add_1�
 model/LSTM/lstm/lstm_cell/Tanh_3Tanh(model/LSTM/lstm/lstm_cell/split:output:3*
T0*(
_output_shapes
:����������2"
 model/LSTM/lstm/lstm_cell/Tanh_3�
 model/LSTM/lstm/lstm_cell/Tanh_4Tanh#model/LSTM/lstm/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:����������2"
 model/LSTM/lstm/lstm_cell/Tanh_4�
model/LSTM/lstm/lstm_cell/mul_2Mul$model/LSTM/lstm/lstm_cell/Tanh_3:y:0$model/LSTM/lstm/lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:����������2!
model/LSTM/lstm/lstm_cell/mul_2�
-model/LSTM/lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   2/
-model/LSTM/lstm/TensorArrayV2_1/element_shape�
model/LSTM/lstm/TensorArrayV2_1TensorListReserve6model/LSTM/lstm/TensorArrayV2_1/element_shape:output:0(model/LSTM/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
model/LSTM/lstm/TensorArrayV2_1n
model/LSTM/lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
model/LSTM/lstm/time�
(model/LSTM/lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2*
(model/LSTM/lstm/while/maximum_iterations�
"model/LSTM/lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"model/LSTM/lstm/while/loop_counter�
model/LSTM/lstm/whileWhile+model/LSTM/lstm/while/loop_counter:output:01model/LSTM/lstm/while/maximum_iterations:output:0model/LSTM/lstm/time:output:0(model/LSTM/lstm/TensorArrayV2_1:handle:0model/LSTM/lstm/zeros:output:0 model/LSTM/lstm/zeros_1:output:0(model/LSTM/lstm/strided_slice_1:output:0Gmodel/LSTM/lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:08model_lstm_lstm_lstm_cell_matmul_readvariableop_resource:model_lstm_lstm_lstm_cell_matmul_1_readvariableop_resource9model_lstm_lstm_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :����������:����������: : : : : *%
_read_only_resource_inputs
	
*.
body&R$
"model_LSTM_lstm_while_body_5286996*.
cond&R$
"model_LSTM_lstm_while_cond_5286995*M
output_shapes<
:: : : : :����������:����������: : : : : *
parallel_iterations 2
model/LSTM/lstm/while�
@model/LSTM/lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   2B
@model/LSTM/lstm/TensorArrayV2Stack/TensorListStack/element_shape�
2model/LSTM/lstm/TensorArrayV2Stack/TensorListStackTensorListStackmodel/LSTM/lstm/while:output:3Imodel/LSTM/lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:d����������*
element_dtype024
2model/LSTM/lstm/TensorArrayV2Stack/TensorListStack�
%model/LSTM/lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2'
%model/LSTM/lstm/strided_slice_3/stack�
'model/LSTM/lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'model/LSTM/lstm/strided_slice_3/stack_1�
'model/LSTM/lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'model/LSTM/lstm/strided_slice_3/stack_2�
model/LSTM/lstm/strided_slice_3StridedSlice;model/LSTM/lstm/TensorArrayV2Stack/TensorListStack:tensor:0.model/LSTM/lstm/strided_slice_3/stack:output:00model/LSTM/lstm/strided_slice_3/stack_1:output:00model/LSTM/lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask2!
model/LSTM/lstm/strided_slice_3�
 model/LSTM/lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 model/LSTM/lstm/transpose_1/perm�
model/LSTM/lstm/transpose_1	Transpose;model/LSTM/lstm/TensorArrayV2Stack/TensorListStack:tensor:0)model/LSTM/lstm/transpose_1/perm:output:0*
T0*,
_output_shapes
:���������d�2
model/LSTM/lstm/transpose_1�
model/LSTM/lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
model/LSTM/lstm/runtime�
.model/tf.math.l2_normalize/l2_normalize/SquareSquare(model/LSTM/lstm/strided_slice_3:output:0*
T0*(
_output_shapes
:����������20
.model/tf.math.l2_normalize/l2_normalize/Square�
=model/tf.math.l2_normalize/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2?
=model/tf.math.l2_normalize/l2_normalize/Sum/reduction_indices�
+model/tf.math.l2_normalize/l2_normalize/SumSum2model/tf.math.l2_normalize/l2_normalize/Square:y:0Fmodel/tf.math.l2_normalize/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(2-
+model/tf.math.l2_normalize/l2_normalize/Sum�
1model/tf.math.l2_normalize/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���.23
1model/tf.math.l2_normalize/l2_normalize/Maximum/y�
/model/tf.math.l2_normalize/l2_normalize/MaximumMaximum4model/tf.math.l2_normalize/l2_normalize/Sum:output:0:model/tf.math.l2_normalize/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:���������21
/model/tf.math.l2_normalize/l2_normalize/Maximum�
-model/tf.math.l2_normalize/l2_normalize/RsqrtRsqrt3model/tf.math.l2_normalize/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:���������2/
-model/tf.math.l2_normalize/l2_normalize/Rsqrt�
'model/tf.math.l2_normalize/l2_normalizeMul(model/LSTM/lstm/strided_slice_3:output:01model/tf.math.l2_normalize/l2_normalize/Rsqrt:y:0*
T0*(
_output_shapes
:����������2)
'model/tf.math.l2_normalize/l2_normalize�
IdentityIdentity+model/tf.math.l2_normalize/l2_normalize:z:01^model/LSTM/lstm/lstm_cell/BiasAdd/ReadVariableOp0^model/LSTM/lstm/lstm_cell/MatMul/ReadVariableOp2^model/LSTM/lstm/lstm_cell/MatMul_1/ReadVariableOp^model/LSTM/lstm/while*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������d:::2d
0model/LSTM/lstm/lstm_cell/BiasAdd/ReadVariableOp0model/LSTM/lstm/lstm_cell/BiasAdd/ReadVariableOp2b
/model/LSTM/lstm/lstm_cell/MatMul/ReadVariableOp/model/LSTM/lstm/lstm_cell/MatMul/ReadVariableOp2f
1model/LSTM/lstm/lstm_cell/MatMul_1/ReadVariableOp1model/LSTM/lstm/lstm_cell/MatMul_1/ReadVariableOp2.
model/LSTM/lstm/whilemodel/LSTM/lstm/while:T P
+
_output_shapes
:���������d
!
_user_specified_name	input_1
�r
�
A__inference_lstm_layer_call_and_return_conditional_losses_5289647
inputs_0,
(lstm_cell_matmul_readvariableop_resource.
*lstm_cell_matmul_1_readvariableop_resource-
)lstm_cell_biasadd_readvariableop_resource
identity��<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp� lstm_cell/BiasAdd/ReadVariableOp�lstm_cell/MatMul/ReadVariableOp�!lstm_cell/MatMul_1/ReadVariableOp�whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:����������2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:����������2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02!
lstm_cell/MatMul/ReadVariableOp�
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell/MatMul�
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp�
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell/MatMul_1�
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell/add�
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp�
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell/BiasAddd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim�
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
lstm_cell/splitu
lstm_cell/TanhTanhlstm_cell/split:output:0*
T0*(
_output_shapes
:����������2
lstm_cell/Tanhy
lstm_cell/Tanh_1Tanhlstm_cell/split:output:1*
T0*(
_output_shapes
:����������2
lstm_cell/Tanh_1�
lstm_cell/mulMullstm_cell/Tanh_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:����������2
lstm_cell/muly
lstm_cell/Tanh_2Tanhlstm_cell/split:output:2*
T0*(
_output_shapes
:����������2
lstm_cell/Tanh_2�
lstm_cell/mul_1Mullstm_cell/Tanh:y:0lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:����������2
lstm_cell/mul_1�
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:����������2
lstm_cell/add_1y
lstm_cell/Tanh_3Tanhlstm_cell/split:output:3*
T0*(
_output_shapes
:����������2
lstm_cell/Tanh_3t
lstm_cell/Tanh_4Tanhlstm_cell/add_1:z:0*
T0*(
_output_shapes
:����������2
lstm_cell/Tanh_4�
lstm_cell/mul_2Mullstm_cell/Tanh_3:y:0lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:����������2
lstm_cell/mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :����������:����������: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_5289550*
condR
while_cond_5289549*M
output_shapes<
:: : : : :����������:����������: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:�������������������*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:�������������������2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square�
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Const�
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum�
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x�
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mul�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp�
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul�
IdentityIdentitystrided_slice_3:output:0=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�
�
__inference_loss_fn_0_5290168I
Elstm_lstm_lstm_cell_kernel_regularizer_square_readvariableop_resource
identity��<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOpElstm_lstm_lstm_cell_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	�*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square�
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Const�
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum�
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x�
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mul�
IdentityIdentity.LSTM/lstm/lstm_cell/kernel/Regularizer/mul:z:0=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp
�7
�
F__inference_lstm_cell_layer_call_and_return_conditional_losses_5287218

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
splitW
TanhTanhsplit:output:0*
T0*(
_output_shapes
:����������2
Tanh[
Tanh_1Tanhsplit:output:1*
T0*(
_output_shapes
:����������2
Tanh_1Z
mulMul
Tanh_1:y:0states_1*
T0*(
_output_shapes
:����������2
mul[
Tanh_2Tanhsplit:output:2*
T0*(
_output_shapes
:����������2
Tanh_2^
mul_1MulTanh:y:0
Tanh_2:y:0*
T0*(
_output_shapes
:����������2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:����������2
add_1[
Tanh_3Tanhsplit:output:3*
T0*(
_output_shapes
:����������2
Tanh_3V
Tanh_4Tanh	add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh_4`
mul_2Mul
Tanh_3:y:0
Tanh_4:y:0*
T0*(
_output_shapes
:����������2
mul_2�
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square�
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Const�
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum�
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x�
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mul�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp�
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul�
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity�

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity_1�

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:���������:����������:����������:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_namestates:PL
(
_output_shapes
:����������
 
_user_specified_namestates
�|
�
A__inference_LSTM_layer_call_and_return_conditional_losses_5289283

inputs1
-lstm_lstm_cell_matmul_readvariableop_resource3
/lstm_lstm_cell_matmul_1_readvariableop_resource2
.lstm_lstm_cell_biasadd_readvariableop_resource
identity��<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp�%lstm/lstm_cell/BiasAdd/ReadVariableOp�$lstm/lstm_cell/MatMul/ReadVariableOp�&lstm/lstm_cell/MatMul_1/ReadVariableOp�
lstm/whileN

lstm/ShapeShapeinputs*
T0*
_output_shapes
:2

lstm/Shape~
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice/stack�
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_1�
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_2�
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_sliceg
lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm/zeros/mul/y�
lstm/zeros/mulMullstm/strided_slice:output:0lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/muli
lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm/zeros/Less/y{
lstm/zeros/LessLesslstm/zeros/mul:z:0lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/Lessm
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
lstm/zeros/packed/1�
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros/packedi
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros/Const�

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*(
_output_shapes
:����������2

lstm/zerosk
lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm/zeros_1/mul/y�
lstm/zeros_1/mulMullstm/strided_slice:output:0lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/mulm
lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm/zeros_1/Less/y�
lstm/zeros_1/LessLesslstm/zeros_1/mul:z:0lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/Lessq
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
lstm/zeros_1/packed/1�
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros_1/packedm
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros_1/Const�
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*(
_output_shapes
:����������2
lstm/zeros_1
lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose/perm�
lstm/transpose	Transposeinputslstm/transpose/perm:output:0*
T0*+
_output_shapes
:d���������2
lstm/transpose^
lstm/Shape_1Shapelstm/transpose:y:0*
T0*
_output_shapes
:2
lstm/Shape_1�
lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_1/stack�
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_1�
lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_2�
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slice_1�
 lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 lstm/TensorArrayV2/element_shape�
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2�
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2<
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shape�
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02.
,lstm/TensorArrayUnstack/TensorListFromTensor�
lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_2/stack�
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_1�
lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_2�
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
lstm/strided_slice_2�
$lstm/lstm_cell/MatMul/ReadVariableOpReadVariableOp-lstm_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02&
$lstm/lstm_cell/MatMul/ReadVariableOp�
lstm/lstm_cell/MatMulMatMullstm/strided_slice_2:output:0,lstm/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm/lstm_cell/MatMul�
&lstm/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp/lstm_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype02(
&lstm/lstm_cell/MatMul_1/ReadVariableOp�
lstm/lstm_cell/MatMul_1MatMullstm/zeros:output:0.lstm/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm/lstm_cell/MatMul_1�
lstm/lstm_cell/addAddV2lstm/lstm_cell/MatMul:product:0!lstm/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm/lstm_cell/add�
%lstm/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp.lstm_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02'
%lstm/lstm_cell/BiasAdd/ReadVariableOp�
lstm/lstm_cell/BiasAddBiasAddlstm/lstm_cell/add:z:0-lstm/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm/lstm_cell/BiasAddn
lstm/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/lstm_cell/Const�
lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm/lstm_cell/split/split_dim�
lstm/lstm_cell/splitSplit'lstm/lstm_cell/split/split_dim:output:0lstm/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
lstm/lstm_cell/split�
lstm/lstm_cell/TanhTanhlstm/lstm_cell/split:output:0*
T0*(
_output_shapes
:����������2
lstm/lstm_cell/Tanh�
lstm/lstm_cell/Tanh_1Tanhlstm/lstm_cell/split:output:1*
T0*(
_output_shapes
:����������2
lstm/lstm_cell/Tanh_1�
lstm/lstm_cell/mulMullstm/lstm_cell/Tanh_1:y:0lstm/zeros_1:output:0*
T0*(
_output_shapes
:����������2
lstm/lstm_cell/mul�
lstm/lstm_cell/Tanh_2Tanhlstm/lstm_cell/split:output:2*
T0*(
_output_shapes
:����������2
lstm/lstm_cell/Tanh_2�
lstm/lstm_cell/mul_1Mullstm/lstm_cell/Tanh:y:0lstm/lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:����������2
lstm/lstm_cell/mul_1�
lstm/lstm_cell/add_1AddV2lstm/lstm_cell/mul:z:0lstm/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:����������2
lstm/lstm_cell/add_1�
lstm/lstm_cell/Tanh_3Tanhlstm/lstm_cell/split:output:3*
T0*(
_output_shapes
:����������2
lstm/lstm_cell/Tanh_3�
lstm/lstm_cell/Tanh_4Tanhlstm/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:����������2
lstm/lstm_cell/Tanh_4�
lstm/lstm_cell/mul_2Mullstm/lstm_cell/Tanh_3:y:0lstm/lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:����������2
lstm/lstm_cell/mul_2�
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   2$
"lstm/TensorArrayV2_1/element_shape�
lstm/TensorArrayV2_1TensorListReserve+lstm/TensorArrayV2_1/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2_1X
	lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
	lstm/time�
lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
lstm/while/maximum_iterationst
lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm/while/loop_counter�

lstm/whileWhile lstm/while/loop_counter:output:0&lstm/while/maximum_iterations:output:0lstm/time:output:0lstm/TensorArrayV2_1:handle:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/strided_slice_1:output:0<lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_lstm_cell_matmul_readvariableop_resource/lstm_lstm_cell_matmul_1_readvariableop_resource.lstm_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :����������:����������: : : : : *%
_read_only_resource_inputs
	
*#
bodyR
lstm_while_body_5289186*#
condR
lstm_while_cond_5289185*M
output_shapes<
:: : : : :����������:����������: : : : : *
parallel_iterations 2

lstm/while�
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   27
5lstm/TensorArrayV2Stack/TensorListStack/element_shape�
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:d����������*
element_dtype02)
'lstm/TensorArrayV2Stack/TensorListStack�
lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
lstm/strided_slice_3/stack�
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_3/stack_1�
lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_3/stack_2�
lstm/strided_slice_3StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask2
lstm/strided_slice_3�
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose_1/perm�
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*,
_output_shapes
:���������d�2
lstm/transpose_1p
lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/runtime�
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp-lstm_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square�
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Const�
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum�
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x�
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mul�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp/lstm_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp�
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul�
IdentityIdentitylstm/strided_slice_3:output:0=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp&^lstm/lstm_cell/BiasAdd/ReadVariableOp%^lstm/lstm_cell/MatMul/ReadVariableOp'^lstm/lstm_cell/MatMul_1/ReadVariableOp^lstm/while*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������d:::2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2N
%lstm/lstm_cell/BiasAdd/ReadVariableOp%lstm/lstm_cell/BiasAdd/ReadVariableOp2L
$lstm/lstm_cell/MatMul/ReadVariableOp$lstm/lstm_cell/MatMul/ReadVariableOp2P
&lstm/lstm_cell/MatMul_1/ReadVariableOp&lstm/lstm_cell/MatMul_1/ReadVariableOp2

lstm/while
lstm/while:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
&__inference_lstm_layer_call_fn_5289658
inputs_0
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_lstm_layer_call_and_return_conditional_losses_52876172
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�
�
'__inference_model_layer_call_fn_5288930

inputs
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_52884962
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������d:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
+__inference_lstm_cell_layer_call_fn_5290140

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:����������:����������:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_cell_layer_call_and_return_conditional_losses_52871732
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity_1�

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:���������:����������:����������:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:RN
(
_output_shapes
:����������
"
_user_specified_name
states/0:RN
(
_output_shapes
:����������
"
_user_specified_name
states/1
�
�
+__inference_lstm_cell_layer_call_fn_5290157

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:����������:����������:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_cell_layer_call_and_return_conditional_losses_52872182
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity_1�

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:���������:����������:����������:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:RN
(
_output_shapes
:����������
"
_user_specified_name
states/0:RN
(
_output_shapes
:����������
"
_user_specified_name
states/1
�@
�
while_body_5288006
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
0while_lstm_cell_matmul_readvariableop_resource_06
2while_lstm_cell_matmul_1_readvariableop_resource_05
1while_lstm_cell_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
.while_lstm_cell_matmul_readvariableop_resource4
0while_lstm_cell_matmul_1_readvariableop_resource3
/while_lstm_cell_biasadd_readvariableop_resource��&while/lstm_cell/BiasAdd/ReadVariableOp�%while/lstm_cell/MatMul/ReadVariableOp�'while/lstm_cell/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02'
%while/lstm_cell/MatMul/ReadVariableOp�
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/MatMul�
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype02)
'while/lstm_cell/MatMul_1/ReadVariableOp�
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/MatMul_1�
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/add�
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02(
&while/lstm_cell/BiasAdd/ReadVariableOp�
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/BiasAddp
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const�
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim�
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
while/lstm_cell/split�
while/lstm_cell/TanhTanhwhile/lstm_cell/split:output:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/Tanh�
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/split:output:1*
T0*(
_output_shapes
:����������2
while/lstm_cell/Tanh_1�
while/lstm_cell/mulMulwhile/lstm_cell/Tanh_1:y:0while_placeholder_3*
T0*(
_output_shapes
:����������2
while/lstm_cell/mul�
while/lstm_cell/Tanh_2Tanhwhile/lstm_cell/split:output:2*
T0*(
_output_shapes
:����������2
while/lstm_cell/Tanh_2�
while/lstm_cell/mul_1Mulwhile/lstm_cell/Tanh:y:0while/lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/mul_1�
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/add_1�
while/lstm_cell/Tanh_3Tanhwhile/lstm_cell/split:output:3*
T0*(
_output_shapes
:����������2
while/lstm_cell/Tanh_3�
while/lstm_cell/Tanh_4Tanhwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/Tanh_4�
while/lstm_cell/mul_2Mulwhile/lstm_cell/Tanh_3:y:0while/lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell/add_1:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :����������:����������: : :::2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: 
� 
�
A__inference_LSTM_layer_call_and_return_conditional_losses_5288146
input_1
lstm_5288126
lstm_5288128
lstm_5288130
identity��<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp�lstm/StatefulPartitionedCall�
lstm/StatefulPartitionedCallStatefulPartitionedCallinput_1lstm_5288126lstm_5288128lstm_5288130*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_lstm_layer_call_and_return_conditional_losses_52879382
lstm/StatefulPartitionedCall�
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_5288126*
_output_shapes
:	�*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square�
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Const�
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum�
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x�
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mul�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_5288128* 
_output_shapes
:
��*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp�
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul�
IdentityIdentity%lstm/StatefulPartitionedCall:output:0=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^lstm/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������d:::2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:T P
+
_output_shapes
:���������d
!
_user_specified_name	input_1
�
�
%__inference_signature_wrapper_5288575
input_1
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__wrapped_model_52870882
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������d:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:���������d
!
_user_specified_name	input_1
� 
�
A__inference_LSTM_layer_call_and_return_conditional_losses_5288169
input_1
lstm_5288149
lstm_5288151
lstm_5288153
identity��<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp�lstm/StatefulPartitionedCall�
lstm/StatefulPartitionedCallStatefulPartitionedCallinput_1lstm_5288149lstm_5288151lstm_5288153*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_lstm_layer_call_and_return_conditional_losses_52881032
lstm/StatefulPartitionedCall�
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_5288149*
_output_shapes
:	�*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square�
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Const�
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum�
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x�
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mul�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_5288151* 
_output_shapes
:
��*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp�
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul�
IdentityIdentity%lstm/StatefulPartitionedCall:output:0=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^lstm/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������d:::2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:T P
+
_output_shapes
:���������d
!
_user_specified_name	input_1
�]
�
A__inference_lstm_layer_call_and_return_conditional_losses_5287617

inputs
lstm_cell_5287523
lstm_cell_5287525
lstm_cell_5287527
identity��<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp�!lstm_cell/StatefulPartitionedCall�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:����������2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:����������2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_5287523lstm_cell_5287525lstm_cell_5287527*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:����������:����������:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_cell_layer_call_and_return_conditional_losses_52871732#
!lstm_cell/StatefulPartitionedCall�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_5287523lstm_cell_5287525lstm_cell_5287527*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :����������:����������: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_5287536*
condR
while_cond_5287535*M
output_shapes<
:: : : : :����������:����������: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:�������������������*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:�������������������2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_5287523*
_output_shapes
:	�*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square�
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Const�
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum�
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x�
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mul�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_5287525* 
_output_shapes
:
��*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp�
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul�
IdentityIdentitystrided_slice_3:output:0=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp"^lstm_cell/StatefulPartitionedCall^while*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�$
�
while_body_5287680
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_5287704_0
while_lstm_cell_5287706_0
while_lstm_cell_5287708_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_5287704
while_lstm_cell_5287706
while_lstm_cell_5287708��'while/lstm_cell/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_5287704_0while_lstm_cell_5287706_0while_lstm_cell_5287708_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:����������:����������:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_cell_layer_call_and_return_conditional_losses_52872182)
'while/lstm_cell/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1(^while/lstm_cell/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2
while/Identity_4�
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2(^while/lstm_cell/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_5287704while_lstm_cell_5287704_0"4
while_lstm_cell_5287706while_lstm_cell_5287706_0"4
while_lstm_cell_5287708while_lstm_cell_5287708_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :����������:����������: : :::2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: 
�$
�
while_body_5287536
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_5287560_0
while_lstm_cell_5287562_0
while_lstm_cell_5287564_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_5287560
while_lstm_cell_5287562
while_lstm_cell_5287564��'while/lstm_cell/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_5287560_0while_lstm_cell_5287562_0while_lstm_cell_5287564_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:����������:����������:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_cell_layer_call_and_return_conditional_losses_52871732)
'while/lstm_cell/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1(^while/lstm_cell/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2
while/Identity_4�
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2(^while/lstm_cell/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_5287560while_lstm_cell_5287560_0"4
while_lstm_cell_5287562while_lstm_cell_5287562_0"4
while_lstm_cell_5287564while_lstm_cell_5287564_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :����������:����������: : :::2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: 
�
�
&__inference_lstm_layer_call_fn_5289669
inputs_0
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_lstm_layer_call_and_return_conditional_losses_52877612
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�,
�
B__inference_model_layer_call_and_return_conditional_losses_5288537

inputs
lstm_5288510
lstm_5288512
lstm_5288514
identity��LSTM/StatefulPartitionedCall�<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp�
LSTM/StatefulPartitionedCallStatefulPartitionedCallinputslstm_5288510lstm_5288512lstm_5288514*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_LSTM_layer_call_and_return_conditional_losses_52881952
LSTM/StatefulPartitionedCall�
(tf.math.l2_normalize/l2_normalize/SquareSquare%LSTM/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:����������2*
(tf.math.l2_normalize/l2_normalize/Square�
7tf.math.l2_normalize/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :29
7tf.math.l2_normalize/l2_normalize/Sum/reduction_indices�
%tf.math.l2_normalize/l2_normalize/SumSum,tf.math.l2_normalize/l2_normalize/Square:y:0@tf.math.l2_normalize/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(2'
%tf.math.l2_normalize/l2_normalize/Sum�
+tf.math.l2_normalize/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���.2-
+tf.math.l2_normalize/l2_normalize/Maximum/y�
)tf.math.l2_normalize/l2_normalize/MaximumMaximum.tf.math.l2_normalize/l2_normalize/Sum:output:04tf.math.l2_normalize/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:���������2+
)tf.math.l2_normalize/l2_normalize/Maximum�
'tf.math.l2_normalize/l2_normalize/RsqrtRsqrt-tf.math.l2_normalize/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:���������2)
'tf.math.l2_normalize/l2_normalize/Rsqrt�
!tf.math.l2_normalize/l2_normalizeMul%LSTM/StatefulPartitionedCall:output:0+tf.math.l2_normalize/l2_normalize/Rsqrt:y:0*
T0*(
_output_shapes
:����������2#
!tf.math.l2_normalize/l2_normalize�
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_5288510*
_output_shapes
:	�*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square�
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Const�
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum�
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x�
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mul�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_5288512* 
_output_shapes
:
��*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp�
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul�
IdentityIdentity%tf.math.l2_normalize/l2_normalize:z:0^LSTM/StatefulPartitionedCall=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������d:::2<
LSTM/StatefulPartitionedCallLSTM/StatefulPartitionedCall2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�	
�
lstm_while_cond_5289185&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3(
$lstm_while_less_lstm_strided_slice_1?
;lstm_while_lstm_while_cond_5289185___redundant_placeholder0?
;lstm_while_lstm_while_cond_5289185___redundant_placeholder1?
;lstm_while_lstm_while_cond_5289185___redundant_placeholder2?
;lstm_while_lstm_while_cond_5289185___redundant_placeholder3
lstm_while_identity
�
lstm/while/LessLesslstm_while_placeholder$lstm_while_less_lstm_strided_slice_1*
T0*
_output_shapes
: 2
lstm/while/Lessl
lstm/while/IdentityIdentitylstm/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm/while/Identity"3
lstm_while_identitylstm/while/Identity:output:0*U
_input_shapesD
B: : : : :����������:����������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
:
�
�
&__inference_lstm_layer_call_fn_5290021

inputs
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_lstm_layer_call_and_return_conditional_losses_52881032
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������d:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�O
�

LSTM_lstm_while_body_52886430
,lstm_lstm_while_lstm_lstm_while_loop_counter6
2lstm_lstm_while_lstm_lstm_while_maximum_iterations
lstm_lstm_while_placeholder!
lstm_lstm_while_placeholder_1!
lstm_lstm_while_placeholder_2!
lstm_lstm_while_placeholder_3/
+lstm_lstm_while_lstm_lstm_strided_slice_1_0k
glstm_lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_lstm_tensorarrayunstack_tensorlistfromtensor_0>
:lstm_lstm_while_lstm_cell_matmul_readvariableop_resource_0@
<lstm_lstm_while_lstm_cell_matmul_1_readvariableop_resource_0?
;lstm_lstm_while_lstm_cell_biasadd_readvariableop_resource_0
lstm_lstm_while_identity
lstm_lstm_while_identity_1
lstm_lstm_while_identity_2
lstm_lstm_while_identity_3
lstm_lstm_while_identity_4
lstm_lstm_while_identity_5-
)lstm_lstm_while_lstm_lstm_strided_slice_1i
elstm_lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_lstm_tensorarrayunstack_tensorlistfromtensor<
8lstm_lstm_while_lstm_cell_matmul_readvariableop_resource>
:lstm_lstm_while_lstm_cell_matmul_1_readvariableop_resource=
9lstm_lstm_while_lstm_cell_biasadd_readvariableop_resource��0LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp�/LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp�1LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp�
ALSTM/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2C
ALSTM/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape�
3LSTM/lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemglstm_lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_lstm_tensorarrayunstack_tensorlistfromtensor_0lstm_lstm_while_placeholderJLSTM/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype025
3LSTM/lstm/while/TensorArrayV2Read/TensorListGetItem�
/LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOpReadVariableOp:lstm_lstm_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype021
/LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp�
 LSTM/lstm/while/lstm_cell/MatMulMatMul:LSTM/lstm/while/TensorArrayV2Read/TensorListGetItem:item:07LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2"
 LSTM/lstm/while/lstm_cell/MatMul�
1LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp<lstm_lstm_while_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype023
1LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp�
"LSTM/lstm/while/lstm_cell/MatMul_1MatMullstm_lstm_while_placeholder_29LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2$
"LSTM/lstm/while/lstm_cell/MatMul_1�
LSTM/lstm/while/lstm_cell/addAddV2*LSTM/lstm/while/lstm_cell/MatMul:product:0,LSTM/lstm/while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
LSTM/lstm/while/lstm_cell/add�
0LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp;lstm_lstm_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype022
0LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp�
!LSTM/lstm/while/lstm_cell/BiasAddBiasAdd!LSTM/lstm/while/lstm_cell/add:z:08LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2#
!LSTM/lstm/while/lstm_cell/BiasAdd�
LSTM/lstm/while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2!
LSTM/lstm/while/lstm_cell/Const�
)LSTM/lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)LSTM/lstm/while/lstm_cell/split/split_dim�
LSTM/lstm/while/lstm_cell/splitSplit2LSTM/lstm/while/lstm_cell/split/split_dim:output:0*LSTM/lstm/while/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2!
LSTM/lstm/while/lstm_cell/split�
LSTM/lstm/while/lstm_cell/TanhTanh(LSTM/lstm/while/lstm_cell/split:output:0*
T0*(
_output_shapes
:����������2 
LSTM/lstm/while/lstm_cell/Tanh�
 LSTM/lstm/while/lstm_cell/Tanh_1Tanh(LSTM/lstm/while/lstm_cell/split:output:1*
T0*(
_output_shapes
:����������2"
 LSTM/lstm/while/lstm_cell/Tanh_1�
LSTM/lstm/while/lstm_cell/mulMul$LSTM/lstm/while/lstm_cell/Tanh_1:y:0lstm_lstm_while_placeholder_3*
T0*(
_output_shapes
:����������2
LSTM/lstm/while/lstm_cell/mul�
 LSTM/lstm/while/lstm_cell/Tanh_2Tanh(LSTM/lstm/while/lstm_cell/split:output:2*
T0*(
_output_shapes
:����������2"
 LSTM/lstm/while/lstm_cell/Tanh_2�
LSTM/lstm/while/lstm_cell/mul_1Mul"LSTM/lstm/while/lstm_cell/Tanh:y:0$LSTM/lstm/while/lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:����������2!
LSTM/lstm/while/lstm_cell/mul_1�
LSTM/lstm/while/lstm_cell/add_1AddV2!LSTM/lstm/while/lstm_cell/mul:z:0#LSTM/lstm/while/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:����������2!
LSTM/lstm/while/lstm_cell/add_1�
 LSTM/lstm/while/lstm_cell/Tanh_3Tanh(LSTM/lstm/while/lstm_cell/split:output:3*
T0*(
_output_shapes
:����������2"
 LSTM/lstm/while/lstm_cell/Tanh_3�
 LSTM/lstm/while/lstm_cell/Tanh_4Tanh#LSTM/lstm/while/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:����������2"
 LSTM/lstm/while/lstm_cell/Tanh_4�
LSTM/lstm/while/lstm_cell/mul_2Mul$LSTM/lstm/while/lstm_cell/Tanh_3:y:0$LSTM/lstm/while/lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:����������2!
LSTM/lstm/while/lstm_cell/mul_2�
4LSTM/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_lstm_while_placeholder_1lstm_lstm_while_placeholder#LSTM/lstm/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype026
4LSTM/lstm/while/TensorArrayV2Write/TensorListSetItemp
LSTM/lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
LSTM/lstm/while/add/y�
LSTM/lstm/while/addAddV2lstm_lstm_while_placeholderLSTM/lstm/while/add/y:output:0*
T0*
_output_shapes
: 2
LSTM/lstm/while/addt
LSTM/lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
LSTM/lstm/while/add_1/y�
LSTM/lstm/while/add_1AddV2,lstm_lstm_while_lstm_lstm_while_loop_counter LSTM/lstm/while/add_1/y:output:0*
T0*
_output_shapes
: 2
LSTM/lstm/while/add_1�
LSTM/lstm/while/IdentityIdentityLSTM/lstm/while/add_1:z:01^LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp0^LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp2^LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
LSTM/lstm/while/Identity�
LSTM/lstm/while/Identity_1Identity2lstm_lstm_while_lstm_lstm_while_maximum_iterations1^LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp0^LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp2^LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
LSTM/lstm/while/Identity_1�
LSTM/lstm/while/Identity_2IdentityLSTM/lstm/while/add:z:01^LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp0^LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp2^LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
LSTM/lstm/while/Identity_2�
LSTM/lstm/while/Identity_3IdentityDLSTM/lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:01^LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp0^LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp2^LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
LSTM/lstm/while/Identity_3�
LSTM/lstm/while/Identity_4Identity#LSTM/lstm/while/lstm_cell/mul_2:z:01^LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp0^LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp2^LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2
LSTM/lstm/while/Identity_4�
LSTM/lstm/while/Identity_5Identity#LSTM/lstm/while/lstm_cell/add_1:z:01^LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp0^LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp2^LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2
LSTM/lstm/while/Identity_5"=
lstm_lstm_while_identity!LSTM/lstm/while/Identity:output:0"A
lstm_lstm_while_identity_1#LSTM/lstm/while/Identity_1:output:0"A
lstm_lstm_while_identity_2#LSTM/lstm/while/Identity_2:output:0"A
lstm_lstm_while_identity_3#LSTM/lstm/while/Identity_3:output:0"A
lstm_lstm_while_identity_4#LSTM/lstm/while/Identity_4:output:0"A
lstm_lstm_while_identity_5#LSTM/lstm/while/Identity_5:output:0"x
9lstm_lstm_while_lstm_cell_biasadd_readvariableop_resource;lstm_lstm_while_lstm_cell_biasadd_readvariableop_resource_0"z
:lstm_lstm_while_lstm_cell_matmul_1_readvariableop_resource<lstm_lstm_while_lstm_cell_matmul_1_readvariableop_resource_0"v
8lstm_lstm_while_lstm_cell_matmul_readvariableop_resource:lstm_lstm_while_lstm_cell_matmul_readvariableop_resource_0"X
)lstm_lstm_while_lstm_lstm_strided_slice_1+lstm_lstm_while_lstm_lstm_strided_slice_1_0"�
elstm_lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_lstm_tensorarrayunstack_tensorlistfromtensorglstm_lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_lstm_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :����������:����������: : :::2d
0LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp0LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp2b
/LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp/LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp2f
1LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp1LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: 
�
�
"model_LSTM_lstm_while_cond_5286995<
8model_lstm_lstm_while_model_lstm_lstm_while_loop_counterB
>model_lstm_lstm_while_model_lstm_lstm_while_maximum_iterations%
!model_lstm_lstm_while_placeholder'
#model_lstm_lstm_while_placeholder_1'
#model_lstm_lstm_while_placeholder_2'
#model_lstm_lstm_while_placeholder_3>
:model_lstm_lstm_while_less_model_lstm_lstm_strided_slice_1U
Qmodel_lstm_lstm_while_model_lstm_lstm_while_cond_5286995___redundant_placeholder0U
Qmodel_lstm_lstm_while_model_lstm_lstm_while_cond_5286995___redundant_placeholder1U
Qmodel_lstm_lstm_while_model_lstm_lstm_while_cond_5286995___redundant_placeholder2U
Qmodel_lstm_lstm_while_model_lstm_lstm_while_cond_5286995___redundant_placeholder3"
model_lstm_lstm_while_identity
�
model/LSTM/lstm/while/LessLess!model_lstm_lstm_while_placeholder:model_lstm_lstm_while_less_model_lstm_lstm_strided_slice_1*
T0*
_output_shapes
: 2
model/LSTM/lstm/while/Less�
model/LSTM/lstm/while/IdentityIdentitymodel/LSTM/lstm/while/Less:z:0*
T0
*
_output_shapes
: 2 
model/LSTM/lstm/while/Identity"I
model_lstm_lstm_while_identity'model/LSTM/lstm/while/Identity:output:0*U
_input_shapesD
B: : : : :����������:����������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
:
�r
�
A__inference_lstm_layer_call_and_return_conditional_losses_5288103

inputs,
(lstm_cell_matmul_readvariableop_resource.
*lstm_cell_matmul_1_readvariableop_resource-
)lstm_cell_biasadd_readvariableop_resource
identity��<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp� lstm_cell/BiasAdd/ReadVariableOp�lstm_cell/MatMul/ReadVariableOp�!lstm_cell/MatMul_1/ReadVariableOp�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:����������2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:����������2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:d���������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02!
lstm_cell/MatMul/ReadVariableOp�
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell/MatMul�
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp�
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell/MatMul_1�
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell/add�
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp�
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell/BiasAddd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim�
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
lstm_cell/splitu
lstm_cell/TanhTanhlstm_cell/split:output:0*
T0*(
_output_shapes
:����������2
lstm_cell/Tanhy
lstm_cell/Tanh_1Tanhlstm_cell/split:output:1*
T0*(
_output_shapes
:����������2
lstm_cell/Tanh_1�
lstm_cell/mulMullstm_cell/Tanh_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:����������2
lstm_cell/muly
lstm_cell/Tanh_2Tanhlstm_cell/split:output:2*
T0*(
_output_shapes
:����������2
lstm_cell/Tanh_2�
lstm_cell/mul_1Mullstm_cell/Tanh:y:0lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:����������2
lstm_cell/mul_1�
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:����������2
lstm_cell/add_1y
lstm_cell/Tanh_3Tanhlstm_cell/split:output:3*
T0*(
_output_shapes
:����������2
lstm_cell/Tanh_3t
lstm_cell/Tanh_4Tanhlstm_cell/add_1:z:0*
T0*(
_output_shapes
:����������2
lstm_cell/Tanh_4�
lstm_cell/mul_2Mullstm_cell/Tanh_3:y:0lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:����������2
lstm_cell/mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :����������:����������: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_5288006*
condR
while_cond_5288005*M
output_shapes<
:: : : : :����������:����������: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:d����������*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:���������d�2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square�
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Const�
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum�
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x�
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mul�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp�
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul�
IdentityIdentitystrided_slice_3:output:0=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������d:::2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�

�
LSTM_lstm_while_cond_52886420
,lstm_lstm_while_lstm_lstm_while_loop_counter6
2lstm_lstm_while_lstm_lstm_while_maximum_iterations
lstm_lstm_while_placeholder!
lstm_lstm_while_placeholder_1!
lstm_lstm_while_placeholder_2!
lstm_lstm_while_placeholder_32
.lstm_lstm_while_less_lstm_lstm_strided_slice_1I
Elstm_lstm_while_lstm_lstm_while_cond_5288642___redundant_placeholder0I
Elstm_lstm_while_lstm_lstm_while_cond_5288642___redundant_placeholder1I
Elstm_lstm_while_lstm_lstm_while_cond_5288642___redundant_placeholder2I
Elstm_lstm_while_lstm_lstm_while_cond_5288642___redundant_placeholder3
lstm_lstm_while_identity
�
LSTM/lstm/while/LessLesslstm_lstm_while_placeholder.lstm_lstm_while_less_lstm_lstm_strided_slice_1*
T0*
_output_shapes
: 2
LSTM/lstm/while/Less{
LSTM/lstm/while/IdentityIdentityLSTM/lstm/while/Less:z:0*
T0
*
_output_shapes
: 2
LSTM/lstm/while/Identity"=
lstm_lstm_while_identity!LSTM/lstm/while/Identity:output:0*U
_input_shapesD
B: : : : :����������:����������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
:
�@
�
while_body_5289737
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
0while_lstm_cell_matmul_readvariableop_resource_06
2while_lstm_cell_matmul_1_readvariableop_resource_05
1while_lstm_cell_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
.while_lstm_cell_matmul_readvariableop_resource4
0while_lstm_cell_matmul_1_readvariableop_resource3
/while_lstm_cell_biasadd_readvariableop_resource��&while/lstm_cell/BiasAdd/ReadVariableOp�%while/lstm_cell/MatMul/ReadVariableOp�'while/lstm_cell/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02'
%while/lstm_cell/MatMul/ReadVariableOp�
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/MatMul�
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype02)
'while/lstm_cell/MatMul_1/ReadVariableOp�
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/MatMul_1�
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/add�
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02(
&while/lstm_cell/BiasAdd/ReadVariableOp�
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/BiasAddp
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const�
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim�
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
while/lstm_cell/split�
while/lstm_cell/TanhTanhwhile/lstm_cell/split:output:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/Tanh�
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/split:output:1*
T0*(
_output_shapes
:����������2
while/lstm_cell/Tanh_1�
while/lstm_cell/mulMulwhile/lstm_cell/Tanh_1:y:0while_placeholder_3*
T0*(
_output_shapes
:����������2
while/lstm_cell/mul�
while/lstm_cell/Tanh_2Tanhwhile/lstm_cell/split:output:2*
T0*(
_output_shapes
:����������2
while/lstm_cell/Tanh_2�
while/lstm_cell/mul_1Mulwhile/lstm_cell/Tanh:y:0while/lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/mul_1�
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/add_1�
while/lstm_cell/Tanh_3Tanhwhile/lstm_cell/split:output:3*
T0*(
_output_shapes
:����������2
while/lstm_cell/Tanh_3�
while/lstm_cell/Tanh_4Tanhwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/Tanh_4�
while/lstm_cell/mul_2Mulwhile/lstm_cell/Tanh_3:y:0while/lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell/add_1:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :����������:����������: : :::2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: 
�|
�
A__inference_LSTM_layer_call_and_return_conditional_losses_5289118

inputs1
-lstm_lstm_cell_matmul_readvariableop_resource3
/lstm_lstm_cell_matmul_1_readvariableop_resource2
.lstm_lstm_cell_biasadd_readvariableop_resource
identity��<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp�%lstm/lstm_cell/BiasAdd/ReadVariableOp�$lstm/lstm_cell/MatMul/ReadVariableOp�&lstm/lstm_cell/MatMul_1/ReadVariableOp�
lstm/whileN

lstm/ShapeShapeinputs*
T0*
_output_shapes
:2

lstm/Shape~
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice/stack�
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_1�
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_2�
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_sliceg
lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm/zeros/mul/y�
lstm/zeros/mulMullstm/strided_slice:output:0lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/muli
lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm/zeros/Less/y{
lstm/zeros/LessLesslstm/zeros/mul:z:0lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/Lessm
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
lstm/zeros/packed/1�
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros/packedi
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros/Const�

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*(
_output_shapes
:����������2

lstm/zerosk
lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm/zeros_1/mul/y�
lstm/zeros_1/mulMullstm/strided_slice:output:0lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/mulm
lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm/zeros_1/Less/y�
lstm/zeros_1/LessLesslstm/zeros_1/mul:z:0lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/Lessq
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
lstm/zeros_1/packed/1�
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros_1/packedm
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros_1/Const�
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*(
_output_shapes
:����������2
lstm/zeros_1
lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose/perm�
lstm/transpose	Transposeinputslstm/transpose/perm:output:0*
T0*+
_output_shapes
:d���������2
lstm/transpose^
lstm/Shape_1Shapelstm/transpose:y:0*
T0*
_output_shapes
:2
lstm/Shape_1�
lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_1/stack�
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_1�
lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_2�
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slice_1�
 lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 lstm/TensorArrayV2/element_shape�
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2�
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2<
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shape�
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02.
,lstm/TensorArrayUnstack/TensorListFromTensor�
lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_2/stack�
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_1�
lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_2�
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
lstm/strided_slice_2�
$lstm/lstm_cell/MatMul/ReadVariableOpReadVariableOp-lstm_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02&
$lstm/lstm_cell/MatMul/ReadVariableOp�
lstm/lstm_cell/MatMulMatMullstm/strided_slice_2:output:0,lstm/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm/lstm_cell/MatMul�
&lstm/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp/lstm_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype02(
&lstm/lstm_cell/MatMul_1/ReadVariableOp�
lstm/lstm_cell/MatMul_1MatMullstm/zeros:output:0.lstm/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm/lstm_cell/MatMul_1�
lstm/lstm_cell/addAddV2lstm/lstm_cell/MatMul:product:0!lstm/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm/lstm_cell/add�
%lstm/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp.lstm_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02'
%lstm/lstm_cell/BiasAdd/ReadVariableOp�
lstm/lstm_cell/BiasAddBiasAddlstm/lstm_cell/add:z:0-lstm/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm/lstm_cell/BiasAddn
lstm/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/lstm_cell/Const�
lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm/lstm_cell/split/split_dim�
lstm/lstm_cell/splitSplit'lstm/lstm_cell/split/split_dim:output:0lstm/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
lstm/lstm_cell/split�
lstm/lstm_cell/TanhTanhlstm/lstm_cell/split:output:0*
T0*(
_output_shapes
:����������2
lstm/lstm_cell/Tanh�
lstm/lstm_cell/Tanh_1Tanhlstm/lstm_cell/split:output:1*
T0*(
_output_shapes
:����������2
lstm/lstm_cell/Tanh_1�
lstm/lstm_cell/mulMullstm/lstm_cell/Tanh_1:y:0lstm/zeros_1:output:0*
T0*(
_output_shapes
:����������2
lstm/lstm_cell/mul�
lstm/lstm_cell/Tanh_2Tanhlstm/lstm_cell/split:output:2*
T0*(
_output_shapes
:����������2
lstm/lstm_cell/Tanh_2�
lstm/lstm_cell/mul_1Mullstm/lstm_cell/Tanh:y:0lstm/lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:����������2
lstm/lstm_cell/mul_1�
lstm/lstm_cell/add_1AddV2lstm/lstm_cell/mul:z:0lstm/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:����������2
lstm/lstm_cell/add_1�
lstm/lstm_cell/Tanh_3Tanhlstm/lstm_cell/split:output:3*
T0*(
_output_shapes
:����������2
lstm/lstm_cell/Tanh_3�
lstm/lstm_cell/Tanh_4Tanhlstm/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:����������2
lstm/lstm_cell/Tanh_4�
lstm/lstm_cell/mul_2Mullstm/lstm_cell/Tanh_3:y:0lstm/lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:����������2
lstm/lstm_cell/mul_2�
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   2$
"lstm/TensorArrayV2_1/element_shape�
lstm/TensorArrayV2_1TensorListReserve+lstm/TensorArrayV2_1/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2_1X
	lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
	lstm/time�
lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
lstm/while/maximum_iterationst
lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm/while/loop_counter�

lstm/whileWhile lstm/while/loop_counter:output:0&lstm/while/maximum_iterations:output:0lstm/time:output:0lstm/TensorArrayV2_1:handle:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/strided_slice_1:output:0<lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0-lstm_lstm_cell_matmul_readvariableop_resource/lstm_lstm_cell_matmul_1_readvariableop_resource.lstm_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :����������:����������: : : : : *%
_read_only_resource_inputs
	
*#
bodyR
lstm_while_body_5289021*#
condR
lstm_while_cond_5289020*M
output_shapes<
:: : : : :����������:����������: : : : : *
parallel_iterations 2

lstm/while�
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   27
5lstm/TensorArrayV2Stack/TensorListStack/element_shape�
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:d����������*
element_dtype02)
'lstm/TensorArrayV2Stack/TensorListStack�
lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
lstm/strided_slice_3/stack�
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_3/stack_1�
lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_3/stack_2�
lstm/strided_slice_3StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask2
lstm/strided_slice_3�
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose_1/perm�
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*,
_output_shapes
:���������d�2
lstm/transpose_1p
lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/runtime�
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp-lstm_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square�
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Const�
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum�
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x�
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mul�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp/lstm_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp�
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul�
IdentityIdentitylstm/strided_slice_3:output:0=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp&^lstm/lstm_cell/BiasAdd/ReadVariableOp%^lstm/lstm_cell/MatMul/ReadVariableOp'^lstm/lstm_cell/MatMul_1/ReadVariableOp^lstm/while*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������d:::2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2N
%lstm/lstm_cell/BiasAdd/ReadVariableOp%lstm/lstm_cell/BiasAdd/ReadVariableOp2L
$lstm/lstm_cell/MatMul/ReadVariableOp$lstm/lstm_cell/MatMul/ReadVariableOp2P
&lstm/lstm_cell/MatMul_1/ReadVariableOp&lstm/lstm_cell/MatMul_1/ReadVariableOp2

lstm/while
lstm/while:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
while_cond_5289736
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_5289736___redundant_placeholder05
1while_while_cond_5289736___redundant_placeholder15
1while_while_cond_5289736___redundant_placeholder25
1while_while_cond_5289736___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :����������:����������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
:
�@
�
while_body_5289550
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
0while_lstm_cell_matmul_readvariableop_resource_06
2while_lstm_cell_matmul_1_readvariableop_resource_05
1while_lstm_cell_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
.while_lstm_cell_matmul_readvariableop_resource4
0while_lstm_cell_matmul_1_readvariableop_resource3
/while_lstm_cell_biasadd_readvariableop_resource��&while/lstm_cell/BiasAdd/ReadVariableOp�%while/lstm_cell/MatMul/ReadVariableOp�'while/lstm_cell/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02'
%while/lstm_cell/MatMul/ReadVariableOp�
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/MatMul�
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype02)
'while/lstm_cell/MatMul_1/ReadVariableOp�
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/MatMul_1�
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/add�
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02(
&while/lstm_cell/BiasAdd/ReadVariableOp�
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/BiasAddp
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const�
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim�
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
while/lstm_cell/split�
while/lstm_cell/TanhTanhwhile/lstm_cell/split:output:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/Tanh�
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/split:output:1*
T0*(
_output_shapes
:����������2
while/lstm_cell/Tanh_1�
while/lstm_cell/mulMulwhile/lstm_cell/Tanh_1:y:0while_placeholder_3*
T0*(
_output_shapes
:����������2
while/lstm_cell/mul�
while/lstm_cell/Tanh_2Tanhwhile/lstm_cell/split:output:2*
T0*(
_output_shapes
:����������2
while/lstm_cell/Tanh_2�
while/lstm_cell/mul_1Mulwhile/lstm_cell/Tanh:y:0while/lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/mul_1�
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/add_1�
while/lstm_cell/Tanh_3Tanhwhile/lstm_cell/split:output:3*
T0*(
_output_shapes
:����������2
while/lstm_cell/Tanh_3�
while/lstm_cell/Tanh_4Tanhwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/Tanh_4�
while/lstm_cell/mul_2Mulwhile/lstm_cell/Tanh_3:y:0while/lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell/add_1:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :����������:����������: : :::2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: 
��
�
B__inference_model_layer_call_and_return_conditional_losses_5288747

inputs6
2lstm_lstm_lstm_cell_matmul_readvariableop_resource8
4lstm_lstm_lstm_cell_matmul_1_readvariableop_resource7
3lstm_lstm_lstm_cell_biasadd_readvariableop_resource
identity��*LSTM/lstm/lstm_cell/BiasAdd/ReadVariableOp�)LSTM/lstm/lstm_cell/MatMul/ReadVariableOp�+LSTM/lstm/lstm_cell/MatMul_1/ReadVariableOp�<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp�LSTM/lstm/whileX
LSTM/lstm/ShapeShapeinputs*
T0*
_output_shapes
:2
LSTM/lstm/Shape�
LSTM/lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
LSTM/lstm/strided_slice/stack�
LSTM/lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
LSTM/lstm/strided_slice/stack_1�
LSTM/lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
LSTM/lstm/strided_slice/stack_2�
LSTM/lstm/strided_sliceStridedSliceLSTM/lstm/Shape:output:0&LSTM/lstm/strided_slice/stack:output:0(LSTM/lstm/strided_slice/stack_1:output:0(LSTM/lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
LSTM/lstm/strided_sliceq
LSTM/lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
LSTM/lstm/zeros/mul/y�
LSTM/lstm/zeros/mulMul LSTM/lstm/strided_slice:output:0LSTM/lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
LSTM/lstm/zeros/muls
LSTM/lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
LSTM/lstm/zeros/Less/y�
LSTM/lstm/zeros/LessLessLSTM/lstm/zeros/mul:z:0LSTM/lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
LSTM/lstm/zeros/Lessw
LSTM/lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
LSTM/lstm/zeros/packed/1�
LSTM/lstm/zeros/packedPack LSTM/lstm/strided_slice:output:0!LSTM/lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
LSTM/lstm/zeros/packeds
LSTM/lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
LSTM/lstm/zeros/Const�
LSTM/lstm/zerosFillLSTM/lstm/zeros/packed:output:0LSTM/lstm/zeros/Const:output:0*
T0*(
_output_shapes
:����������2
LSTM/lstm/zerosu
LSTM/lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
LSTM/lstm/zeros_1/mul/y�
LSTM/lstm/zeros_1/mulMul LSTM/lstm/strided_slice:output:0 LSTM/lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
LSTM/lstm/zeros_1/mulw
LSTM/lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
LSTM/lstm/zeros_1/Less/y�
LSTM/lstm/zeros_1/LessLessLSTM/lstm/zeros_1/mul:z:0!LSTM/lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
LSTM/lstm/zeros_1/Less{
LSTM/lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
LSTM/lstm/zeros_1/packed/1�
LSTM/lstm/zeros_1/packedPack LSTM/lstm/strided_slice:output:0#LSTM/lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
LSTM/lstm/zeros_1/packedw
LSTM/lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
LSTM/lstm/zeros_1/Const�
LSTM/lstm/zeros_1Fill!LSTM/lstm/zeros_1/packed:output:0 LSTM/lstm/zeros_1/Const:output:0*
T0*(
_output_shapes
:����������2
LSTM/lstm/zeros_1�
LSTM/lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
LSTM/lstm/transpose/perm�
LSTM/lstm/transpose	Transposeinputs!LSTM/lstm/transpose/perm:output:0*
T0*+
_output_shapes
:d���������2
LSTM/lstm/transposem
LSTM/lstm/Shape_1ShapeLSTM/lstm/transpose:y:0*
T0*
_output_shapes
:2
LSTM/lstm/Shape_1�
LSTM/lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
LSTM/lstm/strided_slice_1/stack�
!LSTM/lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!LSTM/lstm/strided_slice_1/stack_1�
!LSTM/lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!LSTM/lstm/strided_slice_1/stack_2�
LSTM/lstm/strided_slice_1StridedSliceLSTM/lstm/Shape_1:output:0(LSTM/lstm/strided_slice_1/stack:output:0*LSTM/lstm/strided_slice_1/stack_1:output:0*LSTM/lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
LSTM/lstm/strided_slice_1�
%LSTM/lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%LSTM/lstm/TensorArrayV2/element_shape�
LSTM/lstm/TensorArrayV2TensorListReserve.LSTM/lstm/TensorArrayV2/element_shape:output:0"LSTM/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
LSTM/lstm/TensorArrayV2�
?LSTM/lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2A
?LSTM/lstm/TensorArrayUnstack/TensorListFromTensor/element_shape�
1LSTM/lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorLSTM/lstm/transpose:y:0HLSTM/lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type023
1LSTM/lstm/TensorArrayUnstack/TensorListFromTensor�
LSTM/lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
LSTM/lstm/strided_slice_2/stack�
!LSTM/lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!LSTM/lstm/strided_slice_2/stack_1�
!LSTM/lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!LSTM/lstm/strided_slice_2/stack_2�
LSTM/lstm/strided_slice_2StridedSliceLSTM/lstm/transpose:y:0(LSTM/lstm/strided_slice_2/stack:output:0*LSTM/lstm/strided_slice_2/stack_1:output:0*LSTM/lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
LSTM/lstm/strided_slice_2�
)LSTM/lstm/lstm_cell/MatMul/ReadVariableOpReadVariableOp2lstm_lstm_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02+
)LSTM/lstm/lstm_cell/MatMul/ReadVariableOp�
LSTM/lstm/lstm_cell/MatMulMatMul"LSTM/lstm/strided_slice_2:output:01LSTM/lstm/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
LSTM/lstm/lstm_cell/MatMul�
+LSTM/lstm/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp4lstm_lstm_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+LSTM/lstm/lstm_cell/MatMul_1/ReadVariableOp�
LSTM/lstm/lstm_cell/MatMul_1MatMulLSTM/lstm/zeros:output:03LSTM/lstm/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
LSTM/lstm/lstm_cell/MatMul_1�
LSTM/lstm/lstm_cell/addAddV2$LSTM/lstm/lstm_cell/MatMul:product:0&LSTM/lstm/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
LSTM/lstm/lstm_cell/add�
*LSTM/lstm/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp3lstm_lstm_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02,
*LSTM/lstm/lstm_cell/BiasAdd/ReadVariableOp�
LSTM/lstm/lstm_cell/BiasAddBiasAddLSTM/lstm/lstm_cell/add:z:02LSTM/lstm/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
LSTM/lstm/lstm_cell/BiasAddx
LSTM/lstm/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
LSTM/lstm/lstm_cell/Const�
#LSTM/lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#LSTM/lstm/lstm_cell/split/split_dim�
LSTM/lstm/lstm_cell/splitSplit,LSTM/lstm/lstm_cell/split/split_dim:output:0$LSTM/lstm/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
LSTM/lstm/lstm_cell/split�
LSTM/lstm/lstm_cell/TanhTanh"LSTM/lstm/lstm_cell/split:output:0*
T0*(
_output_shapes
:����������2
LSTM/lstm/lstm_cell/Tanh�
LSTM/lstm/lstm_cell/Tanh_1Tanh"LSTM/lstm/lstm_cell/split:output:1*
T0*(
_output_shapes
:����������2
LSTM/lstm/lstm_cell/Tanh_1�
LSTM/lstm/lstm_cell/mulMulLSTM/lstm/lstm_cell/Tanh_1:y:0LSTM/lstm/zeros_1:output:0*
T0*(
_output_shapes
:����������2
LSTM/lstm/lstm_cell/mul�
LSTM/lstm/lstm_cell/Tanh_2Tanh"LSTM/lstm/lstm_cell/split:output:2*
T0*(
_output_shapes
:����������2
LSTM/lstm/lstm_cell/Tanh_2�
LSTM/lstm/lstm_cell/mul_1MulLSTM/lstm/lstm_cell/Tanh:y:0LSTM/lstm/lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:����������2
LSTM/lstm/lstm_cell/mul_1�
LSTM/lstm/lstm_cell/add_1AddV2LSTM/lstm/lstm_cell/mul:z:0LSTM/lstm/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:����������2
LSTM/lstm/lstm_cell/add_1�
LSTM/lstm/lstm_cell/Tanh_3Tanh"LSTM/lstm/lstm_cell/split:output:3*
T0*(
_output_shapes
:����������2
LSTM/lstm/lstm_cell/Tanh_3�
LSTM/lstm/lstm_cell/Tanh_4TanhLSTM/lstm/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:����������2
LSTM/lstm/lstm_cell/Tanh_4�
LSTM/lstm/lstm_cell/mul_2MulLSTM/lstm/lstm_cell/Tanh_3:y:0LSTM/lstm/lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:����������2
LSTM/lstm/lstm_cell/mul_2�
'LSTM/lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   2)
'LSTM/lstm/TensorArrayV2_1/element_shape�
LSTM/lstm/TensorArrayV2_1TensorListReserve0LSTM/lstm/TensorArrayV2_1/element_shape:output:0"LSTM/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
LSTM/lstm/TensorArrayV2_1b
LSTM/lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
LSTM/lstm/time�
"LSTM/lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2$
"LSTM/lstm/while/maximum_iterations~
LSTM/lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
LSTM/lstm/while/loop_counter�
LSTM/lstm/whileWhile%LSTM/lstm/while/loop_counter:output:0+LSTM/lstm/while/maximum_iterations:output:0LSTM/lstm/time:output:0"LSTM/lstm/TensorArrayV2_1:handle:0LSTM/lstm/zeros:output:0LSTM/lstm/zeros_1:output:0"LSTM/lstm/strided_slice_1:output:0ALSTM/lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:02lstm_lstm_lstm_cell_matmul_readvariableop_resource4lstm_lstm_lstm_cell_matmul_1_readvariableop_resource3lstm_lstm_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :����������:����������: : : : : *%
_read_only_resource_inputs
	
*(
body R
LSTM_lstm_while_body_5288643*(
cond R
LSTM_lstm_while_cond_5288642*M
output_shapes<
:: : : : :����������:����������: : : : : *
parallel_iterations 2
LSTM/lstm/while�
:LSTM/lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   2<
:LSTM/lstm/TensorArrayV2Stack/TensorListStack/element_shape�
,LSTM/lstm/TensorArrayV2Stack/TensorListStackTensorListStackLSTM/lstm/while:output:3CLSTM/lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:d����������*
element_dtype02.
,LSTM/lstm/TensorArrayV2Stack/TensorListStack�
LSTM/lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2!
LSTM/lstm/strided_slice_3/stack�
!LSTM/lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!LSTM/lstm/strided_slice_3/stack_1�
!LSTM/lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!LSTM/lstm/strided_slice_3/stack_2�
LSTM/lstm/strided_slice_3StridedSlice5LSTM/lstm/TensorArrayV2Stack/TensorListStack:tensor:0(LSTM/lstm/strided_slice_3/stack:output:0*LSTM/lstm/strided_slice_3/stack_1:output:0*LSTM/lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask2
LSTM/lstm/strided_slice_3�
LSTM/lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
LSTM/lstm/transpose_1/perm�
LSTM/lstm/transpose_1	Transpose5LSTM/lstm/TensorArrayV2Stack/TensorListStack:tensor:0#LSTM/lstm/transpose_1/perm:output:0*
T0*,
_output_shapes
:���������d�2
LSTM/lstm/transpose_1z
LSTM/lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
LSTM/lstm/runtime�
(tf.math.l2_normalize/l2_normalize/SquareSquare"LSTM/lstm/strided_slice_3:output:0*
T0*(
_output_shapes
:����������2*
(tf.math.l2_normalize/l2_normalize/Square�
7tf.math.l2_normalize/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :29
7tf.math.l2_normalize/l2_normalize/Sum/reduction_indices�
%tf.math.l2_normalize/l2_normalize/SumSum,tf.math.l2_normalize/l2_normalize/Square:y:0@tf.math.l2_normalize/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(2'
%tf.math.l2_normalize/l2_normalize/Sum�
+tf.math.l2_normalize/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���.2-
+tf.math.l2_normalize/l2_normalize/Maximum/y�
)tf.math.l2_normalize/l2_normalize/MaximumMaximum.tf.math.l2_normalize/l2_normalize/Sum:output:04tf.math.l2_normalize/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:���������2+
)tf.math.l2_normalize/l2_normalize/Maximum�
'tf.math.l2_normalize/l2_normalize/RsqrtRsqrt-tf.math.l2_normalize/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:���������2)
'tf.math.l2_normalize/l2_normalize/Rsqrt�
!tf.math.l2_normalize/l2_normalizeMul"LSTM/lstm/strided_slice_3:output:0+tf.math.l2_normalize/l2_normalize/Rsqrt:y:0*
T0*(
_output_shapes
:����������2#
!tf.math.l2_normalize/l2_normalize�
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp2lstm_lstm_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square�
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Const�
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum�
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x�
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mul�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp4lstm_lstm_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp�
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul�
IdentityIdentity%tf.math.l2_normalize/l2_normalize:z:0+^LSTM/lstm/lstm_cell/BiasAdd/ReadVariableOp*^LSTM/lstm/lstm_cell/MatMul/ReadVariableOp,^LSTM/lstm/lstm_cell/MatMul_1/ReadVariableOp=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^LSTM/lstm/while*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������d:::2X
*LSTM/lstm/lstm_cell/BiasAdd/ReadVariableOp*LSTM/lstm/lstm_cell/BiasAdd/ReadVariableOp2V
)LSTM/lstm/lstm_cell/MatMul/ReadVariableOp)LSTM/lstm/lstm_cell/MatMul/ReadVariableOp2Z
+LSTM/lstm/lstm_cell/MatMul_1/ReadVariableOp+LSTM/lstm/lstm_cell/MatMul_1/ReadVariableOp2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2"
LSTM/lstm/whileLSTM/lstm/while:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�G
�	
lstm_while_body_5288286&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3%
!lstm_while_lstm_strided_slice_1_0a
]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_09
5lstm_while_lstm_cell_matmul_readvariableop_resource_0;
7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0:
6lstm_while_lstm_cell_biasadd_readvariableop_resource_0
lstm_while_identity
lstm_while_identity_1
lstm_while_identity_2
lstm_while_identity_3
lstm_while_identity_4
lstm_while_identity_5#
lstm_while_lstm_strided_slice_1_
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor7
3lstm_while_lstm_cell_matmul_readvariableop_resource9
5lstm_while_lstm_cell_matmul_1_readvariableop_resource8
4lstm_while_lstm_cell_biasadd_readvariableop_resource��+lstm/while/lstm_cell/BiasAdd/ReadVariableOp�*lstm/while/lstm_cell/MatMul/ReadVariableOp�,lstm/while/lstm_cell/MatMul_1/ReadVariableOp�
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2>
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape�
.lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0lstm_while_placeholderElstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype020
.lstm/while/TensorArrayV2Read/TensorListGetItem�
*lstm/while/lstm_cell/MatMul/ReadVariableOpReadVariableOp5lstm_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02,
*lstm/while/lstm_cell/MatMul/ReadVariableOp�
lstm/while/lstm_cell/MatMulMatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:02lstm/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm/while/lstm_cell/MatMul�
,lstm/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype02.
,lstm/while/lstm_cell/MatMul_1/ReadVariableOp�
lstm/while/lstm_cell/MatMul_1MatMullstm_while_placeholder_24lstm/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm/while/lstm_cell/MatMul_1�
lstm/while/lstm_cell/addAddV2%lstm/while/lstm_cell/MatMul:product:0'lstm/while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm/while/lstm_cell/add�
+lstm/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp6lstm_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02-
+lstm/while/lstm_cell/BiasAdd/ReadVariableOp�
lstm/while/lstm_cell/BiasAddBiasAddlstm/while/lstm_cell/add:z:03lstm/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm/while/lstm_cell/BiasAddz
lstm/while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/lstm_cell/Const�
$lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm/while/lstm_cell/split/split_dim�
lstm/while/lstm_cell/splitSplit-lstm/while/lstm_cell/split/split_dim:output:0%lstm/while/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
lstm/while/lstm_cell/split�
lstm/while/lstm_cell/TanhTanh#lstm/while/lstm_cell/split:output:0*
T0*(
_output_shapes
:����������2
lstm/while/lstm_cell/Tanh�
lstm/while/lstm_cell/Tanh_1Tanh#lstm/while/lstm_cell/split:output:1*
T0*(
_output_shapes
:����������2
lstm/while/lstm_cell/Tanh_1�
lstm/while/lstm_cell/mulMullstm/while/lstm_cell/Tanh_1:y:0lstm_while_placeholder_3*
T0*(
_output_shapes
:����������2
lstm/while/lstm_cell/mul�
lstm/while/lstm_cell/Tanh_2Tanh#lstm/while/lstm_cell/split:output:2*
T0*(
_output_shapes
:����������2
lstm/while/lstm_cell/Tanh_2�
lstm/while/lstm_cell/mul_1Mullstm/while/lstm_cell/Tanh:y:0lstm/while/lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:����������2
lstm/while/lstm_cell/mul_1�
lstm/while/lstm_cell/add_1AddV2lstm/while/lstm_cell/mul:z:0lstm/while/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:����������2
lstm/while/lstm_cell/add_1�
lstm/while/lstm_cell/Tanh_3Tanh#lstm/while/lstm_cell/split:output:3*
T0*(
_output_shapes
:����������2
lstm/while/lstm_cell/Tanh_3�
lstm/while/lstm_cell/Tanh_4Tanhlstm/while/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:����������2
lstm/while/lstm_cell/Tanh_4�
lstm/while/lstm_cell/mul_2Mullstm/while/lstm_cell/Tanh_3:y:0lstm/while/lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:����������2
lstm/while/lstm_cell/mul_2�
/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_while_placeholder_1lstm_while_placeholderlstm/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype021
/lstm/while/TensorArrayV2Write/TensorListSetItemf
lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/add/y}
lstm/while/addAddV2lstm_while_placeholderlstm/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm/while/addj
lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/add_1/y�
lstm/while/add_1AddV2"lstm_while_lstm_while_loop_counterlstm/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm/while/add_1�
lstm/while/IdentityIdentitylstm/while/add_1:z:0,^lstm/while/lstm_cell/BiasAdd/ReadVariableOp+^lstm/while/lstm_cell/MatMul/ReadVariableOp-^lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm/while/Identity�
lstm/while/Identity_1Identity(lstm_while_lstm_while_maximum_iterations,^lstm/while/lstm_cell/BiasAdd/ReadVariableOp+^lstm/while/lstm_cell/MatMul/ReadVariableOp-^lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm/while/Identity_1�
lstm/while/Identity_2Identitylstm/while/add:z:0,^lstm/while/lstm_cell/BiasAdd/ReadVariableOp+^lstm/while/lstm_cell/MatMul/ReadVariableOp-^lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm/while/Identity_2�
lstm/while/Identity_3Identity?lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0,^lstm/while/lstm_cell/BiasAdd/ReadVariableOp+^lstm/while/lstm_cell/MatMul/ReadVariableOp-^lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm/while/Identity_3�
lstm/while/Identity_4Identitylstm/while/lstm_cell/mul_2:z:0,^lstm/while/lstm_cell/BiasAdd/ReadVariableOp+^lstm/while/lstm_cell/MatMul/ReadVariableOp-^lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2
lstm/while/Identity_4�
lstm/while/Identity_5Identitylstm/while/lstm_cell/add_1:z:0,^lstm/while/lstm_cell/BiasAdd/ReadVariableOp+^lstm/while/lstm_cell/MatMul/ReadVariableOp-^lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2
lstm/while/Identity_5"3
lstm_while_identitylstm/while/Identity:output:0"7
lstm_while_identity_1lstm/while/Identity_1:output:0"7
lstm_while_identity_2lstm/while/Identity_2:output:0"7
lstm_while_identity_3lstm/while/Identity_3:output:0"7
lstm_while_identity_4lstm/while/Identity_4:output:0"7
lstm_while_identity_5lstm/while/Identity_5:output:0"n
4lstm_while_lstm_cell_biasadd_readvariableop_resource6lstm_while_lstm_cell_biasadd_readvariableop_resource_0"p
5lstm_while_lstm_cell_matmul_1_readvariableop_resource7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0"l
3lstm_while_lstm_cell_matmul_readvariableop_resource5lstm_while_lstm_cell_matmul_readvariableop_resource_0"D
lstm_while_lstm_strided_slice_1!lstm_while_lstm_strided_slice_1_0"�
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :����������:����������: : :::2Z
+lstm/while/lstm_cell/BiasAdd/ReadVariableOp+lstm/while/lstm_cell/BiasAdd/ReadVariableOp2X
*lstm/while/lstm_cell/MatMul/ReadVariableOp*lstm/while/lstm_cell/MatMul/ReadVariableOp2\
,lstm/while/lstm_cell/MatMul_1/ReadVariableOp,lstm/while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_5289549
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_5289549___redundant_placeholder05
1while_while_cond_5289549___redundant_placeholder15
1while_while_cond_5289549___redundant_placeholder25
1while_while_cond_5289549___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :����������:����������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
:
�,
�
B__inference_model_layer_call_and_return_conditional_losses_5288433
input_1
lstm_5288406
lstm_5288408
lstm_5288410
identity��LSTM/StatefulPartitionedCall�<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp�
LSTM/StatefulPartitionedCallStatefulPartitionedCallinput_1lstm_5288406lstm_5288408lstm_5288410*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_LSTM_layer_call_and_return_conditional_losses_52883832
LSTM/StatefulPartitionedCall�
(tf.math.l2_normalize/l2_normalize/SquareSquare%LSTM/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:����������2*
(tf.math.l2_normalize/l2_normalize/Square�
7tf.math.l2_normalize/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :29
7tf.math.l2_normalize/l2_normalize/Sum/reduction_indices�
%tf.math.l2_normalize/l2_normalize/SumSum,tf.math.l2_normalize/l2_normalize/Square:y:0@tf.math.l2_normalize/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(2'
%tf.math.l2_normalize/l2_normalize/Sum�
+tf.math.l2_normalize/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���.2-
+tf.math.l2_normalize/l2_normalize/Maximum/y�
)tf.math.l2_normalize/l2_normalize/MaximumMaximum.tf.math.l2_normalize/l2_normalize/Sum:output:04tf.math.l2_normalize/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:���������2+
)tf.math.l2_normalize/l2_normalize/Maximum�
'tf.math.l2_normalize/l2_normalize/RsqrtRsqrt-tf.math.l2_normalize/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:���������2)
'tf.math.l2_normalize/l2_normalize/Rsqrt�
!tf.math.l2_normalize/l2_normalizeMul%LSTM/StatefulPartitionedCall:output:0+tf.math.l2_normalize/l2_normalize/Rsqrt:y:0*
T0*(
_output_shapes
:����������2#
!tf.math.l2_normalize/l2_normalize�
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_5288406*
_output_shapes
:	�*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square�
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Const�
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum�
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x�
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mul�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_5288408* 
_output_shapes
:
��*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp�
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul�
IdentityIdentity%tf.math.l2_normalize/l2_normalize:z:0^LSTM/StatefulPartitionedCall=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������d:::2<
LSTM/StatefulPartitionedCallLSTM/StatefulPartitionedCall2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:T P
+
_output_shapes
:���������d
!
_user_specified_name	input_1
�r
�
A__inference_lstm_layer_call_and_return_conditional_losses_5287938

inputs,
(lstm_cell_matmul_readvariableop_resource.
*lstm_cell_matmul_1_readvariableop_resource-
)lstm_cell_biasadd_readvariableop_resource
identity��<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp� lstm_cell/BiasAdd/ReadVariableOp�lstm_cell/MatMul/ReadVariableOp�!lstm_cell/MatMul_1/ReadVariableOp�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:����������2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:����������2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:d���������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02!
lstm_cell/MatMul/ReadVariableOp�
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell/MatMul�
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp�
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell/MatMul_1�
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell/add�
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp�
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell/BiasAddd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim�
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
lstm_cell/splitu
lstm_cell/TanhTanhlstm_cell/split:output:0*
T0*(
_output_shapes
:����������2
lstm_cell/Tanhy
lstm_cell/Tanh_1Tanhlstm_cell/split:output:1*
T0*(
_output_shapes
:����������2
lstm_cell/Tanh_1�
lstm_cell/mulMullstm_cell/Tanh_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:����������2
lstm_cell/muly
lstm_cell/Tanh_2Tanhlstm_cell/split:output:2*
T0*(
_output_shapes
:����������2
lstm_cell/Tanh_2�
lstm_cell/mul_1Mullstm_cell/Tanh:y:0lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:����������2
lstm_cell/mul_1�
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:����������2
lstm_cell/add_1y
lstm_cell/Tanh_3Tanhlstm_cell/split:output:3*
T0*(
_output_shapes
:����������2
lstm_cell/Tanh_3t
lstm_cell/Tanh_4Tanhlstm_cell/add_1:z:0*
T0*(
_output_shapes
:����������2
lstm_cell/Tanh_4�
lstm_cell/mul_2Mullstm_cell/Tanh_3:y:0lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:����������2
lstm_cell/mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :����������:����������: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_5287841*
condR
while_cond_5287840*M
output_shapes<
:: : : : :����������:����������: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:d����������*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:���������d�2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square�
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Const�
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum�
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x�
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mul�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp�
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul�
IdentityIdentitystrided_slice_3:output:0=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������d:::2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
&__inference_LSTM_layer_call_fn_5289305

inputs
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_LSTM_layer_call_and_return_conditional_losses_52881952
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������d:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�7
�
F__inference_lstm_cell_layer_call_and_return_conditional_losses_5290123

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
splitW
TanhTanhsplit:output:0*
T0*(
_output_shapes
:����������2
Tanh[
Tanh_1Tanhsplit:output:1*
T0*(
_output_shapes
:����������2
Tanh_1Z
mulMul
Tanh_1:y:0states_1*
T0*(
_output_shapes
:����������2
mul[
Tanh_2Tanhsplit:output:2*
T0*(
_output_shapes
:����������2
Tanh_2^
mul_1MulTanh:y:0
Tanh_2:y:0*
T0*(
_output_shapes
:����������2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:����������2
add_1[
Tanh_3Tanhsplit:output:3*
T0*(
_output_shapes
:����������2
Tanh_3V
Tanh_4Tanh	add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh_4`
mul_2Mul
Tanh_3:y:0
Tanh_4:y:0*
T0*(
_output_shapes
:����������2
mul_2�
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square�
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Const�
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum�
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x�
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mul�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp�
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul�
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity�

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity_1�

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:���������:����������:����������:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:RN
(
_output_shapes
:����������
"
_user_specified_name
states/0:RN
(
_output_shapes
:����������
"
_user_specified_name
states/1
�	
�
lstm_while_cond_5288285&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3(
$lstm_while_less_lstm_strided_slice_1?
;lstm_while_lstm_while_cond_5288285___redundant_placeholder0?
;lstm_while_lstm_while_cond_5288285___redundant_placeholder1?
;lstm_while_lstm_while_cond_5288285___redundant_placeholder2?
;lstm_while_lstm_while_cond_5288285___redundant_placeholder3
lstm_while_identity
�
lstm/while/LessLesslstm_while_placeholder$lstm_while_less_lstm_strided_slice_1*
T0*
_output_shapes
: 2
lstm/while/Lessl
lstm/while/IdentityIdentitylstm/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm/while/Identity"3
lstm_while_identitylstm/while/Identity:output:0*U
_input_shapesD
B: : : : :����������:����������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
:
�]
�
A__inference_lstm_layer_call_and_return_conditional_losses_5287761

inputs
lstm_cell_5287667
lstm_cell_5287669
lstm_cell_5287671
identity��<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp�!lstm_cell/StatefulPartitionedCall�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:����������2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:����������2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_5287667lstm_cell_5287669lstm_cell_5287671*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:����������:����������:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_lstm_cell_layer_call_and_return_conditional_losses_52872182#
!lstm_cell/StatefulPartitionedCall�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_5287667lstm_cell_5287669lstm_cell_5287671*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :����������:����������: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_5287680*
condR
while_cond_5287679*M
output_shapes<
:: : : : :����������:����������: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:�������������������*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:�������������������2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_5287667*
_output_shapes
:	�*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square�
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Const�
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum�
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x�
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mul�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_5287669* 
_output_shapes
:
��*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp�
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul�
IdentityIdentitystrided_slice_3:output:0=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp"^lstm_cell/StatefulPartitionedCall^while*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�X
�
"model_LSTM_lstm_while_body_5286996<
8model_lstm_lstm_while_model_lstm_lstm_while_loop_counterB
>model_lstm_lstm_while_model_lstm_lstm_while_maximum_iterations%
!model_lstm_lstm_while_placeholder'
#model_lstm_lstm_while_placeholder_1'
#model_lstm_lstm_while_placeholder_2'
#model_lstm_lstm_while_placeholder_3;
7model_lstm_lstm_while_model_lstm_lstm_strided_slice_1_0w
smodel_lstm_lstm_while_tensorarrayv2read_tensorlistgetitem_model_lstm_lstm_tensorarrayunstack_tensorlistfromtensor_0D
@model_lstm_lstm_while_lstm_cell_matmul_readvariableop_resource_0F
Bmodel_lstm_lstm_while_lstm_cell_matmul_1_readvariableop_resource_0E
Amodel_lstm_lstm_while_lstm_cell_biasadd_readvariableop_resource_0"
model_lstm_lstm_while_identity$
 model_lstm_lstm_while_identity_1$
 model_lstm_lstm_while_identity_2$
 model_lstm_lstm_while_identity_3$
 model_lstm_lstm_while_identity_4$
 model_lstm_lstm_while_identity_59
5model_lstm_lstm_while_model_lstm_lstm_strided_slice_1u
qmodel_lstm_lstm_while_tensorarrayv2read_tensorlistgetitem_model_lstm_lstm_tensorarrayunstack_tensorlistfromtensorB
>model_lstm_lstm_while_lstm_cell_matmul_readvariableop_resourceD
@model_lstm_lstm_while_lstm_cell_matmul_1_readvariableop_resourceC
?model_lstm_lstm_while_lstm_cell_biasadd_readvariableop_resource��6model/LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp�5model/LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp�7model/LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp�
Gmodel/LSTM/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2I
Gmodel/LSTM/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape�
9model/LSTM/lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsmodel_lstm_lstm_while_tensorarrayv2read_tensorlistgetitem_model_lstm_lstm_tensorarrayunstack_tensorlistfromtensor_0!model_lstm_lstm_while_placeholderPmodel/LSTM/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02;
9model/LSTM/lstm/while/TensorArrayV2Read/TensorListGetItem�
5model/LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOpReadVariableOp@model_lstm_lstm_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype027
5model/LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp�
&model/LSTM/lstm/while/lstm_cell/MatMulMatMul@model/LSTM/lstm/while/TensorArrayV2Read/TensorListGetItem:item:0=model/LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2(
&model/LSTM/lstm/while/lstm_cell/MatMul�
7model/LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpBmodel_lstm_lstm_while_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype029
7model/LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp�
(model/LSTM/lstm/while/lstm_cell/MatMul_1MatMul#model_lstm_lstm_while_placeholder_2?model/LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2*
(model/LSTM/lstm/while/lstm_cell/MatMul_1�
#model/LSTM/lstm/while/lstm_cell/addAddV20model/LSTM/lstm/while/lstm_cell/MatMul:product:02model/LSTM/lstm/while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:����������2%
#model/LSTM/lstm/while/lstm_cell/add�
6model/LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpAmodel_lstm_lstm_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype028
6model/LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp�
'model/LSTM/lstm/while/lstm_cell/BiasAddBiasAdd'model/LSTM/lstm/while/lstm_cell/add:z:0>model/LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2)
'model/LSTM/lstm/while/lstm_cell/BiasAdd�
%model/LSTM/lstm/while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2'
%model/LSTM/lstm/while/lstm_cell/Const�
/model/LSTM/lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/model/LSTM/lstm/while/lstm_cell/split/split_dim�
%model/LSTM/lstm/while/lstm_cell/splitSplit8model/LSTM/lstm/while/lstm_cell/split/split_dim:output:00model/LSTM/lstm/while/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2'
%model/LSTM/lstm/while/lstm_cell/split�
$model/LSTM/lstm/while/lstm_cell/TanhTanh.model/LSTM/lstm/while/lstm_cell/split:output:0*
T0*(
_output_shapes
:����������2&
$model/LSTM/lstm/while/lstm_cell/Tanh�
&model/LSTM/lstm/while/lstm_cell/Tanh_1Tanh.model/LSTM/lstm/while/lstm_cell/split:output:1*
T0*(
_output_shapes
:����������2(
&model/LSTM/lstm/while/lstm_cell/Tanh_1�
#model/LSTM/lstm/while/lstm_cell/mulMul*model/LSTM/lstm/while/lstm_cell/Tanh_1:y:0#model_lstm_lstm_while_placeholder_3*
T0*(
_output_shapes
:����������2%
#model/LSTM/lstm/while/lstm_cell/mul�
&model/LSTM/lstm/while/lstm_cell/Tanh_2Tanh.model/LSTM/lstm/while/lstm_cell/split:output:2*
T0*(
_output_shapes
:����������2(
&model/LSTM/lstm/while/lstm_cell/Tanh_2�
%model/LSTM/lstm/while/lstm_cell/mul_1Mul(model/LSTM/lstm/while/lstm_cell/Tanh:y:0*model/LSTM/lstm/while/lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:����������2'
%model/LSTM/lstm/while/lstm_cell/mul_1�
%model/LSTM/lstm/while/lstm_cell/add_1AddV2'model/LSTM/lstm/while/lstm_cell/mul:z:0)model/LSTM/lstm/while/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:����������2'
%model/LSTM/lstm/while/lstm_cell/add_1�
&model/LSTM/lstm/while/lstm_cell/Tanh_3Tanh.model/LSTM/lstm/while/lstm_cell/split:output:3*
T0*(
_output_shapes
:����������2(
&model/LSTM/lstm/while/lstm_cell/Tanh_3�
&model/LSTM/lstm/while/lstm_cell/Tanh_4Tanh)model/LSTM/lstm/while/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:����������2(
&model/LSTM/lstm/while/lstm_cell/Tanh_4�
%model/LSTM/lstm/while/lstm_cell/mul_2Mul*model/LSTM/lstm/while/lstm_cell/Tanh_3:y:0*model/LSTM/lstm/while/lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:����������2'
%model/LSTM/lstm/while/lstm_cell/mul_2�
:model/LSTM/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#model_lstm_lstm_while_placeholder_1!model_lstm_lstm_while_placeholder)model/LSTM/lstm/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02<
:model/LSTM/lstm/while/TensorArrayV2Write/TensorListSetItem|
model/LSTM/lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
model/LSTM/lstm/while/add/y�
model/LSTM/lstm/while/addAddV2!model_lstm_lstm_while_placeholder$model/LSTM/lstm/while/add/y:output:0*
T0*
_output_shapes
: 2
model/LSTM/lstm/while/add�
model/LSTM/lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
model/LSTM/lstm/while/add_1/y�
model/LSTM/lstm/while/add_1AddV28model_lstm_lstm_while_model_lstm_lstm_while_loop_counter&model/LSTM/lstm/while/add_1/y:output:0*
T0*
_output_shapes
: 2
model/LSTM/lstm/while/add_1�
model/LSTM/lstm/while/IdentityIdentitymodel/LSTM/lstm/while/add_1:z:07^model/LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp6^model/LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp8^model/LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2 
model/LSTM/lstm/while/Identity�
 model/LSTM/lstm/while/Identity_1Identity>model_lstm_lstm_while_model_lstm_lstm_while_maximum_iterations7^model/LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp6^model/LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp8^model/LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2"
 model/LSTM/lstm/while/Identity_1�
 model/LSTM/lstm/while/Identity_2Identitymodel/LSTM/lstm/while/add:z:07^model/LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp6^model/LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp8^model/LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2"
 model/LSTM/lstm/while/Identity_2�
 model/LSTM/lstm/while/Identity_3IdentityJmodel/LSTM/lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:07^model/LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp6^model/LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp8^model/LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2"
 model/LSTM/lstm/while/Identity_3�
 model/LSTM/lstm/while/Identity_4Identity)model/LSTM/lstm/while/lstm_cell/mul_2:z:07^model/LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp6^model/LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp8^model/LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2"
 model/LSTM/lstm/while/Identity_4�
 model/LSTM/lstm/while/Identity_5Identity)model/LSTM/lstm/while/lstm_cell/add_1:z:07^model/LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp6^model/LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp8^model/LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2"
 model/LSTM/lstm/while/Identity_5"I
model_lstm_lstm_while_identity'model/LSTM/lstm/while/Identity:output:0"M
 model_lstm_lstm_while_identity_1)model/LSTM/lstm/while/Identity_1:output:0"M
 model_lstm_lstm_while_identity_2)model/LSTM/lstm/while/Identity_2:output:0"M
 model_lstm_lstm_while_identity_3)model/LSTM/lstm/while/Identity_3:output:0"M
 model_lstm_lstm_while_identity_4)model/LSTM/lstm/while/Identity_4:output:0"M
 model_lstm_lstm_while_identity_5)model/LSTM/lstm/while/Identity_5:output:0"�
?model_lstm_lstm_while_lstm_cell_biasadd_readvariableop_resourceAmodel_lstm_lstm_while_lstm_cell_biasadd_readvariableop_resource_0"�
@model_lstm_lstm_while_lstm_cell_matmul_1_readvariableop_resourceBmodel_lstm_lstm_while_lstm_cell_matmul_1_readvariableop_resource_0"�
>model_lstm_lstm_while_lstm_cell_matmul_readvariableop_resource@model_lstm_lstm_while_lstm_cell_matmul_readvariableop_resource_0"p
5model_lstm_lstm_while_model_lstm_lstm_strided_slice_17model_lstm_lstm_while_model_lstm_lstm_strided_slice_1_0"�
qmodel_lstm_lstm_while_tensorarrayv2read_tensorlistgetitem_model_lstm_lstm_tensorarrayunstack_tensorlistfromtensorsmodel_lstm_lstm_while_tensorarrayv2read_tensorlistgetitem_model_lstm_lstm_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :����������:����������: : :::2p
6model/LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp6model/LSTM/lstm/while/lstm_cell/BiasAdd/ReadVariableOp2n
5model/LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp5model/LSTM/lstm/while/lstm_cell/MatMul/ReadVariableOp2r
7model/LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp7model/LSTM/lstm/while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: 
�G
�	
lstm_while_body_5289021&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3%
!lstm_while_lstm_strided_slice_1_0a
]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_09
5lstm_while_lstm_cell_matmul_readvariableop_resource_0;
7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0:
6lstm_while_lstm_cell_biasadd_readvariableop_resource_0
lstm_while_identity
lstm_while_identity_1
lstm_while_identity_2
lstm_while_identity_3
lstm_while_identity_4
lstm_while_identity_5#
lstm_while_lstm_strided_slice_1_
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor7
3lstm_while_lstm_cell_matmul_readvariableop_resource9
5lstm_while_lstm_cell_matmul_1_readvariableop_resource8
4lstm_while_lstm_cell_biasadd_readvariableop_resource��+lstm/while/lstm_cell/BiasAdd/ReadVariableOp�*lstm/while/lstm_cell/MatMul/ReadVariableOp�,lstm/while/lstm_cell/MatMul_1/ReadVariableOp�
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2>
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape�
.lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0lstm_while_placeholderElstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype020
.lstm/while/TensorArrayV2Read/TensorListGetItem�
*lstm/while/lstm_cell/MatMul/ReadVariableOpReadVariableOp5lstm_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02,
*lstm/while/lstm_cell/MatMul/ReadVariableOp�
lstm/while/lstm_cell/MatMulMatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:02lstm/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm/while/lstm_cell/MatMul�
,lstm/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype02.
,lstm/while/lstm_cell/MatMul_1/ReadVariableOp�
lstm/while/lstm_cell/MatMul_1MatMullstm_while_placeholder_24lstm/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm/while/lstm_cell/MatMul_1�
lstm/while/lstm_cell/addAddV2%lstm/while/lstm_cell/MatMul:product:0'lstm/while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm/while/lstm_cell/add�
+lstm/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp6lstm_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02-
+lstm/while/lstm_cell/BiasAdd/ReadVariableOp�
lstm/while/lstm_cell/BiasAddBiasAddlstm/while/lstm_cell/add:z:03lstm/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm/while/lstm_cell/BiasAddz
lstm/while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/lstm_cell/Const�
$lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm/while/lstm_cell/split/split_dim�
lstm/while/lstm_cell/splitSplit-lstm/while/lstm_cell/split/split_dim:output:0%lstm/while/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
lstm/while/lstm_cell/split�
lstm/while/lstm_cell/TanhTanh#lstm/while/lstm_cell/split:output:0*
T0*(
_output_shapes
:����������2
lstm/while/lstm_cell/Tanh�
lstm/while/lstm_cell/Tanh_1Tanh#lstm/while/lstm_cell/split:output:1*
T0*(
_output_shapes
:����������2
lstm/while/lstm_cell/Tanh_1�
lstm/while/lstm_cell/mulMullstm/while/lstm_cell/Tanh_1:y:0lstm_while_placeholder_3*
T0*(
_output_shapes
:����������2
lstm/while/lstm_cell/mul�
lstm/while/lstm_cell/Tanh_2Tanh#lstm/while/lstm_cell/split:output:2*
T0*(
_output_shapes
:����������2
lstm/while/lstm_cell/Tanh_2�
lstm/while/lstm_cell/mul_1Mullstm/while/lstm_cell/Tanh:y:0lstm/while/lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:����������2
lstm/while/lstm_cell/mul_1�
lstm/while/lstm_cell/add_1AddV2lstm/while/lstm_cell/mul:z:0lstm/while/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:����������2
lstm/while/lstm_cell/add_1�
lstm/while/lstm_cell/Tanh_3Tanh#lstm/while/lstm_cell/split:output:3*
T0*(
_output_shapes
:����������2
lstm/while/lstm_cell/Tanh_3�
lstm/while/lstm_cell/Tanh_4Tanhlstm/while/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:����������2
lstm/while/lstm_cell/Tanh_4�
lstm/while/lstm_cell/mul_2Mullstm/while/lstm_cell/Tanh_3:y:0lstm/while/lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:����������2
lstm/while/lstm_cell/mul_2�
/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_while_placeholder_1lstm_while_placeholderlstm/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype021
/lstm/while/TensorArrayV2Write/TensorListSetItemf
lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/add/y}
lstm/while/addAddV2lstm_while_placeholderlstm/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm/while/addj
lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/add_1/y�
lstm/while/add_1AddV2"lstm_while_lstm_while_loop_counterlstm/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm/while/add_1�
lstm/while/IdentityIdentitylstm/while/add_1:z:0,^lstm/while/lstm_cell/BiasAdd/ReadVariableOp+^lstm/while/lstm_cell/MatMul/ReadVariableOp-^lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm/while/Identity�
lstm/while/Identity_1Identity(lstm_while_lstm_while_maximum_iterations,^lstm/while/lstm_cell/BiasAdd/ReadVariableOp+^lstm/while/lstm_cell/MatMul/ReadVariableOp-^lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm/while/Identity_1�
lstm/while/Identity_2Identitylstm/while/add:z:0,^lstm/while/lstm_cell/BiasAdd/ReadVariableOp+^lstm/while/lstm_cell/MatMul/ReadVariableOp-^lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm/while/Identity_2�
lstm/while/Identity_3Identity?lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0,^lstm/while/lstm_cell/BiasAdd/ReadVariableOp+^lstm/while/lstm_cell/MatMul/ReadVariableOp-^lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
lstm/while/Identity_3�
lstm/while/Identity_4Identitylstm/while/lstm_cell/mul_2:z:0,^lstm/while/lstm_cell/BiasAdd/ReadVariableOp+^lstm/while/lstm_cell/MatMul/ReadVariableOp-^lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2
lstm/while/Identity_4�
lstm/while/Identity_5Identitylstm/while/lstm_cell/add_1:z:0,^lstm/while/lstm_cell/BiasAdd/ReadVariableOp+^lstm/while/lstm_cell/MatMul/ReadVariableOp-^lstm/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2
lstm/while/Identity_5"3
lstm_while_identitylstm/while/Identity:output:0"7
lstm_while_identity_1lstm/while/Identity_1:output:0"7
lstm_while_identity_2lstm/while/Identity_2:output:0"7
lstm_while_identity_3lstm/while/Identity_3:output:0"7
lstm_while_identity_4lstm/while/Identity_4:output:0"7
lstm_while_identity_5lstm/while/Identity_5:output:0"n
4lstm_while_lstm_cell_biasadd_readvariableop_resource6lstm_while_lstm_cell_biasadd_readvariableop_resource_0"p
5lstm_while_lstm_cell_matmul_1_readvariableop_resource7lstm_while_lstm_cell_matmul_1_readvariableop_resource_0"l
3lstm_while_lstm_cell_matmul_readvariableop_resource5lstm_while_lstm_cell_matmul_readvariableop_resource_0"D
lstm_while_lstm_strided_slice_1!lstm_while_lstm_strided_slice_1_0"�
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :����������:����������: : :::2Z
+lstm/while/lstm_cell/BiasAdd/ReadVariableOp+lstm/while/lstm_cell/BiasAdd/ReadVariableOp2X
*lstm/while/lstm_cell/MatMul/ReadVariableOp*lstm/while/lstm_cell/MatMul/ReadVariableOp2\
,lstm/while/lstm_cell/MatMul_1/ReadVariableOp,lstm/while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_5288005
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_5288005___redundant_placeholder05
1while_while_cond_5288005___redundant_placeholder15
1while_while_cond_5288005___redundant_placeholder25
1while_while_cond_5288005___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :����������:����������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_5287840
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_5287840___redundant_placeholder05
1while_while_cond_5287840___redundant_placeholder15
1while_while_cond_5287840___redundant_placeholder25
1while_while_cond_5287840___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :����������:����������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
:
�r
�
A__inference_lstm_layer_call_and_return_conditional_losses_5289834

inputs,
(lstm_cell_matmul_readvariableop_resource.
*lstm_cell_matmul_1_readvariableop_resource-
)lstm_cell_biasadd_readvariableop_resource
identity��<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp� lstm_cell/BiasAdd/ReadVariableOp�lstm_cell/MatMul/ReadVariableOp�!lstm_cell/MatMul_1/ReadVariableOp�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:����������2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:����������2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:d���������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02!
lstm_cell/MatMul/ReadVariableOp�
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell/MatMul�
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp�
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell/MatMul_1�
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell/add�
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp�
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell/BiasAddd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim�
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
lstm_cell/splitu
lstm_cell/TanhTanhlstm_cell/split:output:0*
T0*(
_output_shapes
:����������2
lstm_cell/Tanhy
lstm_cell/Tanh_1Tanhlstm_cell/split:output:1*
T0*(
_output_shapes
:����������2
lstm_cell/Tanh_1�
lstm_cell/mulMullstm_cell/Tanh_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:����������2
lstm_cell/muly
lstm_cell/Tanh_2Tanhlstm_cell/split:output:2*
T0*(
_output_shapes
:����������2
lstm_cell/Tanh_2�
lstm_cell/mul_1Mullstm_cell/Tanh:y:0lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:����������2
lstm_cell/mul_1�
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:����������2
lstm_cell/add_1y
lstm_cell/Tanh_3Tanhlstm_cell/split:output:3*
T0*(
_output_shapes
:����������2
lstm_cell/Tanh_3t
lstm_cell/Tanh_4Tanhlstm_cell/add_1:z:0*
T0*(
_output_shapes
:����������2
lstm_cell/Tanh_4�
lstm_cell/mul_2Mullstm_cell/Tanh_3:y:0lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:����������2
lstm_cell/mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :����������:����������: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_5289737*
condR
while_cond_5289736*M
output_shapes<
:: : : : :����������:����������: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:d����������*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:���������d�2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square�
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Const�
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum�
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x�
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mul�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp�
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul�
IdentityIdentitystrided_slice_3:output:0=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������d:::2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
&__inference_lstm_layer_call_fn_5290010

inputs
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_lstm_layer_call_and_return_conditional_losses_52879382
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������d:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
while_cond_5287679
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_5287679___redundant_placeholder05
1while_while_cond_5287679___redundant_placeholder15
1while_while_cond_5287679___redundant_placeholder25
1while_while_cond_5287679___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :����������:����������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
:
� 
�
A__inference_LSTM_layer_call_and_return_conditional_losses_5288195

inputs
lstm_5288175
lstm_5288177
lstm_5288179
identity��<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp�lstm/StatefulPartitionedCall�
lstm/StatefulPartitionedCallStatefulPartitionedCallinputslstm_5288175lstm_5288177lstm_5288179*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_lstm_layer_call_and_return_conditional_losses_52881032
lstm/StatefulPartitionedCall�
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_5288175*
_output_shapes
:	�*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square�
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Const�
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum�
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x�
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mul�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_5288177* 
_output_shapes
:
��*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp�
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul�
IdentityIdentity%lstm/StatefulPartitionedCall:output:0=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^lstm/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������d:::2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�7
�
F__inference_lstm_cell_layer_call_and_return_conditional_losses_5290078

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
splitW
TanhTanhsplit:output:0*
T0*(
_output_shapes
:����������2
Tanh[
Tanh_1Tanhsplit:output:1*
T0*(
_output_shapes
:����������2
Tanh_1Z
mulMul
Tanh_1:y:0states_1*
T0*(
_output_shapes
:����������2
mul[
Tanh_2Tanhsplit:output:2*
T0*(
_output_shapes
:����������2
Tanh_2^
mul_1MulTanh:y:0
Tanh_2:y:0*
T0*(
_output_shapes
:����������2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:����������2
add_1[
Tanh_3Tanhsplit:output:3*
T0*(
_output_shapes
:����������2
Tanh_3V
Tanh_4Tanh	add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh_4`
mul_2Mul
Tanh_3:y:0
Tanh_4:y:0*
T0*(
_output_shapes
:����������2
mul_2�
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square�
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Const�
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum�
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x�
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mul�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp�
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul�
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity�

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity_1�

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:���������:����������:����������:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:RN
(
_output_shapes
:����������
"
_user_specified_name
states/0:RN
(
_output_shapes
:����������
"
_user_specified_name
states/1
�	
�
lstm_while_cond_5289020&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3(
$lstm_while_less_lstm_strided_slice_1?
;lstm_while_lstm_while_cond_5289020___redundant_placeholder0?
;lstm_while_lstm_while_cond_5289020___redundant_placeholder1?
;lstm_while_lstm_while_cond_5289020___redundant_placeholder2?
;lstm_while_lstm_while_cond_5289020___redundant_placeholder3
lstm_while_identity
�
lstm/while/LessLesslstm_while_placeholder$lstm_while_less_lstm_strided_slice_1*
T0*
_output_shapes
: 2
lstm/while/Lessl
lstm/while/IdentityIdentitylstm/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm/while/Identity"3
lstm_while_identitylstm/while/Identity:output:0*U
_input_shapesD
B: : : : :����������:����������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
:
�7
�
F__inference_lstm_cell_layer_call_and_return_conditional_losses_5287173

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
splitW
TanhTanhsplit:output:0*
T0*(
_output_shapes
:����������2
Tanh[
Tanh_1Tanhsplit:output:1*
T0*(
_output_shapes
:����������2
Tanh_1Z
mulMul
Tanh_1:y:0states_1*
T0*(
_output_shapes
:����������2
mul[
Tanh_2Tanhsplit:output:2*
T0*(
_output_shapes
:����������2
Tanh_2^
mul_1MulTanh:y:0
Tanh_2:y:0*
T0*(
_output_shapes
:����������2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:����������2
add_1[
Tanh_3Tanhsplit:output:3*
T0*(
_output_shapes
:����������2
Tanh_3V
Tanh_4Tanh	add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh_4`
mul_2Mul
Tanh_3:y:0
Tanh_4:y:0*
T0*(
_output_shapes
:����������2
mul_2�
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square�
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Const�
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum�
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x�
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mul�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp�
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul�
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity�

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity_1�

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*Z
_input_shapesI
G:���������:����������:����������:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_namestates:PL
(
_output_shapes
:����������
 
_user_specified_namestates
�
�
&__inference_LSTM_layer_call_fn_5288204
input_1
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_LSTM_layer_call_and_return_conditional_losses_52881952
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������d:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:���������d
!
_user_specified_name	input_1
�,
�
B__inference_model_layer_call_and_return_conditional_losses_5288496

inputs
lstm_5288469
lstm_5288471
lstm_5288473
identity��LSTM/StatefulPartitionedCall�<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp�
LSTM/StatefulPartitionedCallStatefulPartitionedCallinputslstm_5288469lstm_5288471lstm_5288473*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_LSTM_layer_call_and_return_conditional_losses_52883832
LSTM/StatefulPartitionedCall�
(tf.math.l2_normalize/l2_normalize/SquareSquare%LSTM/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:����������2*
(tf.math.l2_normalize/l2_normalize/Square�
7tf.math.l2_normalize/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :29
7tf.math.l2_normalize/l2_normalize/Sum/reduction_indices�
%tf.math.l2_normalize/l2_normalize/SumSum,tf.math.l2_normalize/l2_normalize/Square:y:0@tf.math.l2_normalize/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(2'
%tf.math.l2_normalize/l2_normalize/Sum�
+tf.math.l2_normalize/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���.2-
+tf.math.l2_normalize/l2_normalize/Maximum/y�
)tf.math.l2_normalize/l2_normalize/MaximumMaximum.tf.math.l2_normalize/l2_normalize/Sum:output:04tf.math.l2_normalize/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:���������2+
)tf.math.l2_normalize/l2_normalize/Maximum�
'tf.math.l2_normalize/l2_normalize/RsqrtRsqrt-tf.math.l2_normalize/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:���������2)
'tf.math.l2_normalize/l2_normalize/Rsqrt�
!tf.math.l2_normalize/l2_normalizeMul%LSTM/StatefulPartitionedCall:output:0+tf.math.l2_normalize/l2_normalize/Rsqrt:y:0*
T0*(
_output_shapes
:����������2#
!tf.math.l2_normalize/l2_normalize�
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_5288469*
_output_shapes
:	�*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square�
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Const�
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum�
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x�
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mul�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_5288471* 
_output_shapes
:
��*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp�
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul�
IdentityIdentity%tf.math.l2_normalize/l2_normalize:z:0^LSTM/StatefulPartitionedCall=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������d:::2<
LSTM/StatefulPartitionedCallLSTM/StatefulPartitionedCall2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�r
�
A__inference_lstm_layer_call_and_return_conditional_losses_5289482
inputs_0,
(lstm_cell_matmul_readvariableop_resource.
*lstm_cell_matmul_1_readvariableop_resource-
)lstm_cell_biasadd_readvariableop_resource
identity��<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp� lstm_cell/BiasAdd/ReadVariableOp�lstm_cell/MatMul/ReadVariableOp�!lstm_cell/MatMul_1/ReadVariableOp�whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:����������2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:����������2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02!
lstm_cell/MatMul/ReadVariableOp�
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell/MatMul�
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp�
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell/MatMul_1�
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell/add�
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp�
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell/BiasAddd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim�
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
lstm_cell/splitu
lstm_cell/TanhTanhlstm_cell/split:output:0*
T0*(
_output_shapes
:����������2
lstm_cell/Tanhy
lstm_cell/Tanh_1Tanhlstm_cell/split:output:1*
T0*(
_output_shapes
:����������2
lstm_cell/Tanh_1�
lstm_cell/mulMullstm_cell/Tanh_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:����������2
lstm_cell/muly
lstm_cell/Tanh_2Tanhlstm_cell/split:output:2*
T0*(
_output_shapes
:����������2
lstm_cell/Tanh_2�
lstm_cell/mul_1Mullstm_cell/Tanh:y:0lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:����������2
lstm_cell/mul_1�
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:����������2
lstm_cell/add_1y
lstm_cell/Tanh_3Tanhlstm_cell/split:output:3*
T0*(
_output_shapes
:����������2
lstm_cell/Tanh_3t
lstm_cell/Tanh_4Tanhlstm_cell/add_1:z:0*
T0*(
_output_shapes
:����������2
lstm_cell/Tanh_4�
lstm_cell/mul_2Mullstm_cell/Tanh_3:y:0lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:����������2
lstm_cell/mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :����������:����������: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_5289385*
condR
while_cond_5289384*M
output_shapes<
:: : : : :����������:����������: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:�������������������*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:�������������������2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime�
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02>
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp�
-LSTM/lstm/lstm_cell/kernel/Regularizer/SquareSquareDLSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2/
-LSTM/lstm/lstm_cell/kernel/Regularizer/Square�
,LSTM/lstm/lstm_cell/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/Const�
*LSTM/lstm/lstm_cell/kernel/Regularizer/SumSum1LSTM/lstm/lstm_cell/kernel/Regularizer/Square:y:05LSTM/lstm/lstm_cell/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/Sum�
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x�
*LSTM/lstm/lstm_cell/kernel/Regularizer/mulMul5LSTM/lstm/lstm_cell/kernel/Regularizer/mul/x:output:03LSTM/lstm/lstm_cell/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*LSTM/lstm/lstm_cell/kernel/Regularizer/mul�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
��*
dtype02H
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp�
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SquareSquareNLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��29
7LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/SumSum;LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square:y:0?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum�
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<28
6LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x�
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mulMul?LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul/x:output:0=LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/mul�
IdentityIdentitystrided_slice_3:output:0=^LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOpG^LSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::2|
<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp<LSTM/lstm/lstm_cell/kernel/Regularizer/Square/ReadVariableOp2�
FLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOpFLSTM/lstm/lstm_cell/recurrent_kernel/Regularizer/Square/ReadVariableOp2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�%
�
 __inference__traced_save_5290235
file_prefix+
'savev2_adagrad_iter_read_readvariableop	,
(savev2_adagrad_decay_read_readvariableop4
0savev2_adagrad_learning_rate_read_readvariableop9
5savev2_lstm_lstm_lstm_cell_kernel_read_readvariableopC
?savev2_lstm_lstm_lstm_cell_recurrent_kernel_read_readvariableop7
3savev2_lstm_lstm_lstm_cell_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopM
Isavev2_adagrad_lstm_lstm_lstm_cell_kernel_accumulator_read_readvariableopW
Ssavev2_adagrad_lstm_lstm_lstm_cell_recurrent_kernel_accumulator_read_readvariableopK
Gsavev2_adagrad_lstm_lstm_lstm_cell_bias_accumulator_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBVtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBVtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEBVtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/accumulator/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*+
value"B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_adagrad_iter_read_readvariableop(savev2_adagrad_decay_read_readvariableop0savev2_adagrad_learning_rate_read_readvariableop5savev2_lstm_lstm_lstm_cell_kernel_read_readvariableop?savev2_lstm_lstm_lstm_cell_recurrent_kernel_read_readvariableop3savev2_lstm_lstm_lstm_cell_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopIsavev2_adagrad_lstm_lstm_lstm_cell_kernel_accumulator_read_readvariableopSsavev2_adagrad_lstm_lstm_lstm_cell_recurrent_kernel_accumulator_read_readvariableopGsavev2_adagrad_lstm_lstm_lstm_cell_bias_accumulator_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*]
_input_shapesL
J: : : : :	�:
��:�: : :	�:
��:�: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:

_output_shapes
: :

_output_shapes
: :%	!

_output_shapes
:	�:&
"
 
_output_shapes
:
��:!

_output_shapes	
:�:

_output_shapes
: 
�
�
'__inference_model_layer_call_fn_5288941

inputs
unknown
	unknown_0
	unknown_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_52885372
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������d:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�@
�
while_body_5287841
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
0while_lstm_cell_matmul_readvariableop_resource_06
2while_lstm_cell_matmul_1_readvariableop_resource_05
1while_lstm_cell_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
.while_lstm_cell_matmul_readvariableop_resource4
0while_lstm_cell_matmul_1_readvariableop_resource3
/while_lstm_cell_biasadd_readvariableop_resource��&while/lstm_cell/BiasAdd/ReadVariableOp�%while/lstm_cell/MatMul/ReadVariableOp�'while/lstm_cell/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02'
%while/lstm_cell/MatMul/ReadVariableOp�
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/MatMul�
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype02)
'while/lstm_cell/MatMul_1/ReadVariableOp�
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/MatMul_1�
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/add�
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02(
&while/lstm_cell/BiasAdd/ReadVariableOp�
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/BiasAddp
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const�
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim�
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
while/lstm_cell/split�
while/lstm_cell/TanhTanhwhile/lstm_cell/split:output:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/Tanh�
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/split:output:1*
T0*(
_output_shapes
:����������2
while/lstm_cell/Tanh_1�
while/lstm_cell/mulMulwhile/lstm_cell/Tanh_1:y:0while_placeholder_3*
T0*(
_output_shapes
:����������2
while/lstm_cell/mul�
while/lstm_cell/Tanh_2Tanhwhile/lstm_cell/split:output:2*
T0*(
_output_shapes
:����������2
while/lstm_cell/Tanh_2�
while/lstm_cell/mul_1Mulwhile/lstm_cell/Tanh:y:0while/lstm_cell/Tanh_2:y:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/mul_1�
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/add_1�
while/lstm_cell/Tanh_3Tanhwhile/lstm_cell/split:output:3*
T0*(
_output_shapes
:����������2
while/lstm_cell/Tanh_3�
while/lstm_cell/Tanh_4Tanhwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/Tanh_4�
while/lstm_cell/mul_2Mulwhile/lstm_cell/Tanh_3:y:0while/lstm_cell/Tanh_4:y:0*
T0*(
_output_shapes
:����������2
while/lstm_cell/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1�
while/IdentityIdentitywhile/add_1:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity�
while/Identity_1Identitywhile_while_maximum_iterations'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1�
while/Identity_2Identitywhile/add:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell/add_1:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :����������:����������: : :::2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:����������:.*
(
_output_shapes
:����������:

_output_shapes
: :

_output_shapes
: "�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
?
input_14
serving_default_input_1:0���������dI
tf.math.l2_normalize1
StatefulPartitionedCall:0����������tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer-2
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api
	
signatures
*=&call_and_return_all_conditional_losses
>__call__
?_default_save_signature"�
_tf_keras_network�{"class_name": "Functional", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Model", "config": {"layer was saved without config": true}, "name": "LSTM", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.l2_normalize", "trainable": true, "dtype": "float32", "function": "math.l2_normalize"}, "name": "tf.math.l2_normalize", "inbound_nodes": [["LSTM", 0, 0, {"axis": 1, "epsilon": 1e-10}]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["tf.math.l2_normalize", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 100, 2]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 2]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}, "training_config": {"loss": "loss", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adagrad", "config": {"name": "Adagrad", "learning_rate": 0.10000000149011612, "decay": 0.0, "initial_accumulator_value": 0.1, "epsilon": 1e-07}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
�

lstm
trainable_variables
regularization_losses
	variables
	keras_api
*@&call_and_return_all_conditional_losses
A__call__"�
_tf_keras_model�{"class_name": "Model", "name": "LSTM", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Model"}}
�
	keras_api"�
_tf_keras_layer�{"class_name": "TFOpLambda", "name": "tf.math.l2_normalize", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.l2_normalize", "trainable": true, "dtype": "float32", "function": "math.l2_normalize"}}
t
iter
	decay
learning_rateaccumulator:accumulator;accumulator<"
	optimizer
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
�
metrics
trainable_variables
non_trainable_variables
layer_metrics
regularization_losses
	variables

layers
layer_regularization_losses
>__call__
?_default_save_signature
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
,
Bserving_default"
signature_map
�
cell

state_spec
trainable_variables
regularization_losses
	variables
 	keras_api
*C&call_and_return_all_conditional_losses
D__call__"�

_tf_keras_rnn_layer�
{"class_name": "LSTM", "name": "lstm", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 128, "activation": "tanh", "recurrent_activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "recurrent_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 2]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 2]}}
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
�
!metrics
trainable_variables
"non_trainable_variables
#layer_metrics
regularization_losses
	variables

$layers
%layer_regularization_losses
A__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
:	 (2Adagrad/iter
: (2Adagrad/decay
: (2Adagrad/learning_rate
-:+	�2LSTM/lstm/lstm_cell/kernel
8:6
��2$LSTM/lstm/lstm_cell/recurrent_kernel
':%�2LSTM/lstm/lstm_cell/bias
'
&0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
�	

kernel
recurrent_kernel
bias
'trainable_variables
(regularization_losses
)	variables
*	keras_api
E__call__
*F&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "LSTMCell", "name": "lstm_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "recurrent_activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "recurrent_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
�

+states
,metrics
trainable_variables
-non_trainable_variables
.layer_metrics
regularization_losses
	variables

/layers
0layer_regularization_losses
D__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'

0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
	1total
	2count
3	variables
4	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
5
0
1
2"
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
�
5metrics
6non_trainable_variables
7layer_metrics
'trainable_variables
(regularization_losses
)	variables

8layers
9layer_regularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
.
10
21"
trackable_list_wrapper
-
3	variables"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
?:=	�2.Adagrad/LSTM/lstm/lstm_cell/kernel/accumulator
J:H
��28Adagrad/LSTM/lstm/lstm_cell/recurrent_kernel/accumulator
9:7�2,Adagrad/LSTM/lstm/lstm_cell/bias/accumulator
�2�
B__inference_model_layer_call_and_return_conditional_losses_5288747
B__inference_model_layer_call_and_return_conditional_losses_5288919
B__inference_model_layer_call_and_return_conditional_losses_5288433
B__inference_model_layer_call_and_return_conditional_losses_5288463�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
'__inference_model_layer_call_fn_5288930
'__inference_model_layer_call_fn_5288505
'__inference_model_layer_call_fn_5288941
'__inference_model_layer_call_fn_5288546�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
"__inference__wrapped_model_5287088�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� **�'
%�"
input_1���������d
�2�
A__inference_LSTM_layer_call_and_return_conditional_losses_5288146
A__inference_LSTM_layer_call_and_return_conditional_losses_5288169
A__inference_LSTM_layer_call_and_return_conditional_losses_5289118
A__inference_LSTM_layer_call_and_return_conditional_losses_5289283�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_LSTM_layer_call_fn_5288215
&__inference_LSTM_layer_call_fn_5289305
&__inference_LSTM_layer_call_fn_5289294
&__inference_LSTM_layer_call_fn_5288204�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_signature_wrapper_5288575input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_lstm_layer_call_and_return_conditional_losses_5289482
A__inference_lstm_layer_call_and_return_conditional_losses_5289834
A__inference_lstm_layer_call_and_return_conditional_losses_5289647
A__inference_lstm_layer_call_and_return_conditional_losses_5289999�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
&__inference_lstm_layer_call_fn_5290010
&__inference_lstm_layer_call_fn_5290021
&__inference_lstm_layer_call_fn_5289669
&__inference_lstm_layer_call_fn_5289658�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
+__inference_lstm_cell_layer_call_fn_5290140
+__inference_lstm_cell_layer_call_fn_5290157�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
F__inference_lstm_cell_layer_call_and_return_conditional_losses_5290078
F__inference_lstm_cell_layer_call_and_return_conditional_losses_5290123�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
__inference_loss_fn_0_5290168�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_1_5290179�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� �
A__inference_LSTM_layer_call_and_return_conditional_losses_5288146g8�5
.�+
%�"
input_1���������d
p
� "&�#
�
0����������
� �
A__inference_LSTM_layer_call_and_return_conditional_losses_5288169g8�5
.�+
%�"
input_1���������d
p 
� "&�#
�
0����������
� �
A__inference_LSTM_layer_call_and_return_conditional_losses_5289118f7�4
-�*
$�!
inputs���������d
p
� "&�#
�
0����������
� �
A__inference_LSTM_layer_call_and_return_conditional_losses_5289283f7�4
-�*
$�!
inputs���������d
p 
� "&�#
�
0����������
� �
&__inference_LSTM_layer_call_fn_5288204Z8�5
.�+
%�"
input_1���������d
p
� "������������
&__inference_LSTM_layer_call_fn_5288215Z8�5
.�+
%�"
input_1���������d
p 
� "������������
&__inference_LSTM_layer_call_fn_5289294Y7�4
-�*
$�!
inputs���������d
p
� "������������
&__inference_LSTM_layer_call_fn_5289305Y7�4
-�*
$�!
inputs���������d
p 
� "������������
"__inference__wrapped_model_5287088�4�1
*�'
%�"
input_1���������d
� "L�I
G
tf.math.l2_normalize/�,
tf.math.l2_normalize����������<
__inference_loss_fn_0_5290168�

� 
� "� <
__inference_loss_fn_1_5290179�

� 
� "� �
F__inference_lstm_cell_layer_call_and_return_conditional_losses_5290078���
x�u
 �
inputs���������
M�J
#� 
states/0����������
#� 
states/1����������
p
� "v�s
l�i
�
0/0����������
G�D
 �
0/1/0����������
 �
0/1/1����������
� �
F__inference_lstm_cell_layer_call_and_return_conditional_losses_5290123���
x�u
 �
inputs���������
M�J
#� 
states/0����������
#� 
states/1����������
p 
� "v�s
l�i
�
0/0����������
G�D
 �
0/1/0����������
 �
0/1/1����������
� �
+__inference_lstm_cell_layer_call_fn_5290140���
x�u
 �
inputs���������
M�J
#� 
states/0����������
#� 
states/1����������
p
� "f�c
�
0����������
C�@
�
1/0����������
�
1/1�����������
+__inference_lstm_cell_layer_call_fn_5290157���
x�u
 �
inputs���������
M�J
#� 
states/0����������
#� 
states/1����������
p 
� "f�c
�
0����������
C�@
�
1/0����������
�
1/1�����������
A__inference_lstm_layer_call_and_return_conditional_losses_5289482~O�L
E�B
4�1
/�,
inputs/0������������������

 
p

 
� "&�#
�
0����������
� �
A__inference_lstm_layer_call_and_return_conditional_losses_5289647~O�L
E�B
4�1
/�,
inputs/0������������������

 
p 

 
� "&�#
�
0����������
� �
A__inference_lstm_layer_call_and_return_conditional_losses_5289834n?�<
5�2
$�!
inputs���������d

 
p

 
� "&�#
�
0����������
� �
A__inference_lstm_layer_call_and_return_conditional_losses_5289999n?�<
5�2
$�!
inputs���������d

 
p 

 
� "&�#
�
0����������
� �
&__inference_lstm_layer_call_fn_5289658qO�L
E�B
4�1
/�,
inputs/0������������������

 
p

 
� "������������
&__inference_lstm_layer_call_fn_5289669qO�L
E�B
4�1
/�,
inputs/0������������������

 
p 

 
� "������������
&__inference_lstm_layer_call_fn_5290010a?�<
5�2
$�!
inputs���������d

 
p

 
� "������������
&__inference_lstm_layer_call_fn_5290021a?�<
5�2
$�!
inputs���������d

 
p 

 
� "������������
B__inference_model_layer_call_and_return_conditional_losses_5288433k<�9
2�/
%�"
input_1���������d
p

 
� "&�#
�
0����������
� �
B__inference_model_layer_call_and_return_conditional_losses_5288463k<�9
2�/
%�"
input_1���������d
p 

 
� "&�#
�
0����������
� �
B__inference_model_layer_call_and_return_conditional_losses_5288747j;�8
1�.
$�!
inputs���������d
p

 
� "&�#
�
0����������
� �
B__inference_model_layer_call_and_return_conditional_losses_5288919j;�8
1�.
$�!
inputs���������d
p 

 
� "&�#
�
0����������
� �
'__inference_model_layer_call_fn_5288505^<�9
2�/
%�"
input_1���������d
p

 
� "������������
'__inference_model_layer_call_fn_5288546^<�9
2�/
%�"
input_1���������d
p 

 
� "������������
'__inference_model_layer_call_fn_5288930];�8
1�.
$�!
inputs���������d
p

 
� "������������
'__inference_model_layer_call_fn_5288941];�8
1�.
$�!
inputs���������d
p 

 
� "������������
%__inference_signature_wrapper_5288575�?�<
� 
5�2
0
input_1%�"
input_1���������d"L�I
G
tf.math.l2_normalize/�,
tf.math.l2_normalize����������