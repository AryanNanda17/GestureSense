├я
Ў╞
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
А
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resourceИ
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
В
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р
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
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
Т
ResizeBilinear
images"T
size
resized_images"
Ttype:
2	"
align_cornersbool( "
half_pixel_centersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
┴
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
executor_typestring Ии
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.13.02v2.13.0-rc2-7-g1cb1a030a628ш│
~
Adam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/v
w
'Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/v*
_output_shapes
:*
dtype0
Ж
Adam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_3/kernel/v

)Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/v*
_output_shapes

:@*
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:@*
dtype0
З
Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А@*&
shared_nameAdam/dense_2/kernel/v
А
)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes
:	А@*
dtype0
А
Adam/conv2d_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_9/bias/v
y
(Adam/conv2d_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/bias/v*
_output_shapes
:@*
dtype0
Р
Adam/conv2d_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_9/kernel/v
Й
*Adam/conv2d_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/kernel/v*&
_output_shapes
:@@*
dtype0
А
Adam/conv2d_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_8/bias/v
y
(Adam/conv2d_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/bias/v*
_output_shapes
:@*
dtype0
Р
Adam/conv2d_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_8/kernel/v
Й
*Adam/conv2d_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/kernel/v*&
_output_shapes
:@@*
dtype0
А
Adam/conv2d_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_7/bias/v
y
(Adam/conv2d_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/bias/v*
_output_shapes
:@*
dtype0
Р
Adam/conv2d_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_7/kernel/v
Й
*Adam/conv2d_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/kernel/v*&
_output_shapes
:@@*
dtype0
А
Adam/conv2d_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_6/bias/v
y
(Adam/conv2d_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/bias/v*
_output_shapes
:@*
dtype0
Р
Adam/conv2d_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv2d_6/kernel/v
Й
*Adam/conv2d_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/kernel/v*&
_output_shapes
: @*
dtype0
А
Adam/conv2d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_5/bias/v
y
(Adam/conv2d_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/v*
_output_shapes
: *
dtype0
Р
Adam/conv2d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_5/kernel/v
Й
*Adam/conv2d_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/v*&
_output_shapes
: *
dtype0
~
Adam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/m
w
'Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/m*
_output_shapes
:*
dtype0
Ж
Adam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_3/kernel/m

)Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/m*
_output_shapes

:@*
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:@*
dtype0
З
Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А@*&
shared_nameAdam/dense_2/kernel/m
А
)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes
:	А@*
dtype0
А
Adam/conv2d_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_9/bias/m
y
(Adam/conv2d_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/bias/m*
_output_shapes
:@*
dtype0
Р
Adam/conv2d_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_9/kernel/m
Й
*Adam/conv2d_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/kernel/m*&
_output_shapes
:@@*
dtype0
А
Adam/conv2d_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_8/bias/m
y
(Adam/conv2d_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/bias/m*
_output_shapes
:@*
dtype0
Р
Adam/conv2d_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_8/kernel/m
Й
*Adam/conv2d_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/kernel/m*&
_output_shapes
:@@*
dtype0
А
Adam/conv2d_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_7/bias/m
y
(Adam/conv2d_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/bias/m*
_output_shapes
:@*
dtype0
Р
Adam/conv2d_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_7/kernel/m
Й
*Adam/conv2d_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/kernel/m*&
_output_shapes
:@@*
dtype0
А
Adam/conv2d_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_6/bias/m
y
(Adam/conv2d_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/bias/m*
_output_shapes
:@*
dtype0
Р
Adam/conv2d_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv2d_6/kernel/m
Й
*Adam/conv2d_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/kernel/m*&
_output_shapes
: @*
dtype0
А
Adam/conv2d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_5/bias/m
y
(Adam/conv2d_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/m*
_output_shapes
: *
dtype0
Р
Adam/conv2d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_5/kernel/m
Й
*Adam/conv2d_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/m*&
_output_shapes
: *
dtype0
И
random_rotation/StateVarVarHandleOp*
_output_shapes
: *
dtype0	*
shape:*)
shared_namerandom_rotation/StateVar
Б
,random_rotation/StateVar/Read/ReadVariableOpReadVariableOprandom_rotation/StateVar*
_output_shapes
:*
dtype0	
А
random_flip/StateVarVarHandleOp*
_output_shapes
: *
dtype0	*
shape:*%
shared_namerandom_flip/StateVar
y
(random_flip/StateVar/Read/ReadVariableOpReadVariableOprandom_flip/StateVar*
_output_shapes
:*
dtype0	
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
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

:@*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:@*
dtype0
y
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А@*
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	А@*
dtype0
r
conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_9/bias
k
!conv2d_9/bias/Read/ReadVariableOpReadVariableOpconv2d_9/bias*
_output_shapes
:@*
dtype0
В
conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_9/kernel
{
#conv2d_9/kernel/Read/ReadVariableOpReadVariableOpconv2d_9/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_8/bias
k
!conv2d_8/bias/Read/ReadVariableOpReadVariableOpconv2d_8/bias*
_output_shapes
:@*
dtype0
В
conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_8/kernel
{
#conv2d_8/kernel/Read/ReadVariableOpReadVariableOpconv2d_8/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_7/bias
k
!conv2d_7/bias/Read/ReadVariableOpReadVariableOpconv2d_7/bias*
_output_shapes
:@*
dtype0
В
conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_7/kernel
{
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_6/bias
k
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes
:@*
dtype0
В
conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2d_6/kernel
{
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*&
_output_shapes
: @*
dtype0
r
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_5/bias
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes
: *
dtype0
В
conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_5/kernel
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
: *
dtype0
Ч
 serving_default_sequential_inputPlaceholder*1
_output_shapes
:         АА*
dtype0*&
shape:         АА
▓
StatefulPartitionedCallStatefulPartitionedCall serving_default_sequential_inputconv2d_5/kernelconv2d_5/biasconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasconv2d_8/kernelconv2d_8/biasconv2d_9/kernelconv2d_9/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference_signature_wrapper_19355

NoOpNoOp
чС
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*бС
valueЦСBТС BКС
╩
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer_with_weights-4
layer-10
layer-11
layer-12
layer_with_weights-5
layer-13
layer_with_weights-6
layer-14
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
и
layer-0
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses* 
к
!layer-0
"layer-1
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses*
╚
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

/kernel
0bias
 1_jit_compiled_convolution_op*
О
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses* 
╚
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses

>kernel
?bias
 @_jit_compiled_convolution_op*
О
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses* 
╚
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses

Mkernel
Nbias
 O_jit_compiled_convolution_op*
О
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses* 
╚
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses

\kernel
]bias
 ^_jit_compiled_convolution_op*
О
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses* 
╚
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses

kkernel
lbias
 m_jit_compiled_convolution_op*
О
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses* 
О
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses* 
и
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses
Аkernel
	Бbias*
о
В	variables
Гtrainable_variables
Дregularization_losses
Е	keras_api
Ж__call__
+З&call_and_return_all_conditional_losses
Иkernel
	Йbias*
n
/0
01
>2
?3
M4
N5
\6
]7
k8
l9
А10
Б11
И12
Й13*
n
/0
01
>2
?3
M4
N5
\6
]7
k8
l9
А10
Б11
И12
Й13*
* 
╡
Кnon_trainable_variables
Лlayers
Мmetrics
 Нlayer_regularization_losses
Оlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Пtrace_0
Рtrace_1* 

Сtrace_0
Тtrace_1* 
* 
щ
	Уiter
Фbeta_1
Хbeta_2

Цdecay
Чlearning_rate/m╧0m╨>m╤?m╥Mm╙Nm╘\m╒]m╓km╫lm╪	Аm┘	Бm┌	Иm█	Йm▄/v▌0v▐>v▀?vрMvсNvт\vу]vфkvхlvц	Аvч	Бvш	Иvщ	Йvъ*

Шserving_default* 
Ф
Щ	variables
Ъtrainable_variables
Ыregularization_losses
Ь	keras_api
Э__call__
+Ю&call_and_return_all_conditional_losses* 
Ф
Я	variables
аtrainable_variables
бregularization_losses
в	keras_api
г__call__
+д&call_and_return_all_conditional_losses* 
* 
* 
* 
Ц
еnon_trainable_variables
жlayers
зmetrics
 иlayer_regularization_losses
йlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses* 

кtrace_0
лtrace_1* 

мtrace_0
нtrace_1* 
о
о	variables
пtrainable_variables
░regularization_losses
▒	keras_api
▓__call__
+│&call_and_return_all_conditional_losses
┤_random_generator*
о
╡	variables
╢trainable_variables
╖regularization_losses
╕	keras_api
╣__call__
+║&call_and_return_all_conditional_losses
╗_random_generator*
* 
* 
* 
Ш
╝non_trainable_variables
╜layers
╛metrics
 ┐layer_regularization_losses
└layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses*

┴trace_0
┬trace_1* 

├trace_0
─trace_1* 

/0
01*

/0
01*
* 
Ш
┼non_trainable_variables
╞layers
╟metrics
 ╚layer_regularization_losses
╔layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*

╩trace_0* 

╦trace_0* 
_Y
VARIABLE_VALUEconv2d_5/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_5/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
╠non_trainable_variables
═layers
╬metrics
 ╧layer_regularization_losses
╨layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses* 

╤trace_0* 

╥trace_0* 

>0
?1*

>0
?1*
* 
Ш
╙non_trainable_variables
╘layers
╒metrics
 ╓layer_regularization_losses
╫layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses*

╪trace_0* 

┘trace_0* 
_Y
VARIABLE_VALUEconv2d_6/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_6/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
┌non_trainable_variables
█layers
▄metrics
 ▌layer_regularization_losses
▐layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses* 

▀trace_0* 

рtrace_0* 

M0
N1*

M0
N1*
* 
Ш
сnon_trainable_variables
тlayers
уmetrics
 фlayer_regularization_losses
хlayer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses*

цtrace_0* 

чtrace_0* 
_Y
VARIABLE_VALUEconv2d_7/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_7/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
шnon_trainable_variables
щlayers
ъmetrics
 ыlayer_regularization_losses
ьlayer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses* 

эtrace_0* 

юtrace_0* 

\0
]1*

\0
]1*
* 
Ш
яnon_trainable_variables
Ёlayers
ёmetrics
 Єlayer_regularization_losses
єlayer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses*

Їtrace_0* 

їtrace_0* 
_Y
VARIABLE_VALUEconv2d_8/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_8/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
Ўnon_trainable_variables
ўlayers
°metrics
 ∙layer_regularization_losses
·layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses* 

√trace_0* 

№trace_0* 

k0
l1*

k0
l1*
* 
Ш
¤non_trainable_variables
■layers
 metrics
 Аlayer_regularization_losses
Бlayer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses*

Вtrace_0* 

Гtrace_0* 
_Y
VARIABLE_VALUEconv2d_9/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_9/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
Дnon_trainable_variables
Еlayers
Жmetrics
 Зlayer_regularization_losses
Иlayer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses* 

Йtrace_0* 

Кtrace_0* 
* 
* 
* 
Ц
Лnon_trainable_variables
Мlayers
Нmetrics
 Оlayer_regularization_losses
Пlayer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses* 

Рtrace_0* 

Сtrace_0* 

А0
Б1*

А0
Б1*
* 
Ш
Тnon_trainable_variables
Уlayers
Фmetrics
 Хlayer_regularization_losses
Цlayer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Чtrace_0* 

Шtrace_0* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

И0
Й1*

И0
Й1*
* 
Ю
Щnon_trainable_variables
Ъlayers
Ыmetrics
 Ьlayer_regularization_losses
Эlayer_metrics
В	variables
Гtrainable_variables
Дregularization_losses
Ж__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses*

Юtrace_0* 

Яtrace_0* 
^X
VARIABLE_VALUEdense_3/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_3/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
r
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14*

а0
б1*
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ь
вnon_trainable_variables
гlayers
дmetrics
 еlayer_regularization_losses
жlayer_metrics
Щ	variables
Ъtrainable_variables
Ыregularization_losses
Э__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses* 

зtrace_0* 

иtrace_0* 
* 
* 
* 
Ь
йnon_trainable_variables
кlayers
лmetrics
 мlayer_regularization_losses
нlayer_metrics
Я	variables
аtrainable_variables
бregularization_losses
г__call__
+д&call_and_return_all_conditional_losses
'д"call_and_return_conditional_losses* 

оtrace_0* 

пtrace_0* 
* 

0
1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
Ь
░non_trainable_variables
▒layers
▓metrics
 │layer_regularization_losses
┤layer_metrics
о	variables
пtrainable_variables
░regularization_losses
▓__call__
+│&call_and_return_all_conditional_losses
'│"call_and_return_conditional_losses* 

╡trace_0
╢trace_1* 

╖trace_0
╕trace_1* 

╣
_generator*
* 
* 
* 
Ь
║non_trainable_variables
╗layers
╝metrics
 ╜layer_regularization_losses
╛layer_metrics
╡	variables
╢trainable_variables
╖regularization_losses
╣__call__
+║&call_and_return_all_conditional_losses
'║"call_and_return_conditional_losses* 

┐trace_0
└trace_1* 

┴trace_0
┬trace_1* 

├
_generator*
* 

!0
"1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
─	variables
┼	keras_api

╞total

╟count*
M
╚	variables
╔	keras_api

╩total

╦count
╠
_fn_kwargs*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

═
_state_var*
* 
* 
* 
* 
* 
* 
* 
* 
* 

╬
_state_var*

╞0
╟1*

─	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

╩0
╦1*

╚	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
Аz
VARIABLE_VALUErandom_flip/StateVarRlayer-1/layer-0/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUErandom_rotation/StateVarRlayer-1/layer-1/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/conv2d_5/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_5/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/conv2d_6/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_6/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/conv2d_7/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_7/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/conv2d_8/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_8/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/conv2d_9/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_9/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUEAdam/dense_3/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_3/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/conv2d_5/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_5/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/conv2d_6/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_6/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/conv2d_7/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_7/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/conv2d_8/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_8/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/conv2d_9/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_9/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUEAdam/dense_3/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_3/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
╝

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameconv2d_5/kernelconv2d_5/biasconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasconv2d_8/kernelconv2d_8/biasconv2d_9/kernelconv2d_9/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountrandom_flip/StateVarrandom_rotation/StateVarAdam/conv2d_5/kernel/mAdam/conv2d_5/bias/mAdam/conv2d_6/kernel/mAdam/conv2d_6/bias/mAdam/conv2d_7/kernel/mAdam/conv2d_7/bias/mAdam/conv2d_8/kernel/mAdam/conv2d_8/bias/mAdam/conv2d_9/kernel/mAdam/conv2d_9/bias/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/dense_3/kernel/mAdam/dense_3/bias/mAdam/conv2d_5/kernel/vAdam/conv2d_5/bias/vAdam/conv2d_6/kernel/vAdam/conv2d_6/bias/vAdam/conv2d_7/kernel/vAdam/conv2d_7/bias/vAdam/conv2d_8/kernel/vAdam/conv2d_8/bias/vAdam/conv2d_9/kernel/vAdam/conv2d_9/bias/vAdam/dense_2/kernel/vAdam/dense_2/bias/vAdam/dense_3/kernel/vAdam/dense_3/bias/vConst*B
Tin;
927*
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
GPU 2J 8В *'
f"R 
__inference__traced_save_20191
╖

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_5/kernelconv2d_5/biasconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasconv2d_8/kernelconv2d_8/biasconv2d_9/kernelconv2d_9/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountrandom_flip/StateVarrandom_rotation/StateVarAdam/conv2d_5/kernel/mAdam/conv2d_5/bias/mAdam/conv2d_6/kernel/mAdam/conv2d_6/bias/mAdam/conv2d_7/kernel/mAdam/conv2d_7/bias/mAdam/conv2d_8/kernel/mAdam/conv2d_8/bias/mAdam/conv2d_9/kernel/mAdam/conv2d_9/bias/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/dense_3/kernel/mAdam/dense_3/bias/mAdam/conv2d_5/kernel/vAdam/conv2d_5/bias/vAdam/conv2d_6/kernel/vAdam/conv2d_6/bias/vAdam/conv2d_7/kernel/vAdam/conv2d_7/bias/vAdam/conv2d_8/kernel/vAdam/conv2d_8/bias/vAdam/conv2d_9/kernel/vAdam/conv2d_9/bias/vAdam/dense_2/kernel/vAdam/dense_2/bias/vAdam/dense_3/kernel/vAdam/dense_3/bias/v*A
Tin:
826*
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
GPU 2J 8В **
f%R#
!__inference__traced_restore_20359ьЯ
Ў
`
D__inference_rescaling_layer_call_and_return_conditional_losses_18626

inputs
identityK
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *БАА;M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    _
mulMulinputsCast/x:output:0*
T0*1
_output_shapes
:         ААd
addAddV2mul:z:0Cast_1/x:output:0*
T0*1
_output_shapes
:         ААY
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:         АА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         АА:Y U
1
_output_shapes
:         АА
 
_user_specified_nameinputs
н
_
C__inference_resizing_layer_call_and_return_conditional_losses_19567

inputs
identity\
resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"А   А   Ы
resize/ResizeBilinearResizeBilinearinputsresize/size:output:0*
T0*1
_output_shapes
:         АА*
half_pixel_centers(x
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*1
_output_shapes
:         АА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         АА:Y U
1
_output_shapes
:         АА
 
_user_specified_nameinputs
┼
E
)__inference_rescaling_layer_call_fn_19572

inputs
identity╣
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_rescaling_layer_call_and_return_conditional_losses_18626j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         АА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         АА:Y U
1
_output_shapes
:         АА
 
_user_specified_nameinputs
Т
Э
(__inference_conv2d_8_layer_call_fn_19454

inputs!
unknown:@@
	unknown_0:@
identityИвStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_19066w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs:%!

_user_specified_name19448:%!

_user_specified_name19450
╔
G
+__inference_random_flip_layer_call_fn_19592

inputs
identity╗
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_random_flip_layer_call_and_return_conditional_losses_18900j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         АА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         АА:Y U
1
_output_shapes
:         АА
 
_user_specified_nameinputs
Т
Э
(__inference_conv2d_9_layer_call_fn_19484

inputs!
unknown:@@
	unknown_0:@
identityИвStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_19083w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs:%!

_user_specified_name19478:%!

_user_specified_name19480
╢
K
/__inference_max_pooling2d_7_layer_call_fn_19440

inputs
identity╪
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_18971Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
│
№
C__inference_conv2d_7_layer_call_and_return_conditional_losses_19435

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ш
f
J__inference_random_rotation_layer_call_and_return_conditional_losses_18905

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:         АА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         АА:Y U
1
_output_shapes
:         АА
 
_user_specified_nameinputs
╔

/__inference_random_rotation_layer_call_fn_19714

inputs
unknown:	
identityИвStatefulPartitionedCall┘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_random_rotation_layer_call_and_return_conditional_losses_18889y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         АА<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         АА: 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         АА
 
_user_specified_nameinputs:%!

_user_specified_name19710
Т
f
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_19445

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
│
№
C__inference_conv2d_9_layer_call_and_return_conditional_losses_19495

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Т	
n
G__inference_sequential_1_layer_call_and_return_conditional_losses_18908
random_flip_input
identity╥
random_flip/PartitionedCallPartitionedCallrandom_flip_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_random_flip_layer_call_and_return_conditional_losses_18900э
random_rotation/PartitionedCallPartitionedCall$random_flip/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_random_rotation_layer_call_and_return_conditional_losses_18905z
IdentityIdentity(random_rotation/PartitionedCall:output:0*
T0*1
_output_shapes
:         АА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         АА:d `
1
_output_shapes
:         АА
+
_user_specified_namerandom_flip_input
Ў
З
G__inference_sequential_1_layer_call_and_return_conditional_losses_18894
random_flip_input
random_flip_18769:	#
random_rotation_18890:	
identityИв#random_flip/StatefulPartitionedCallв'random_rotation/StatefulPartitionedCallЎ
#random_flip/StatefulPartitionedCallStatefulPartitionedCallrandom_flip_inputrandom_flip_18769*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_random_flip_layer_call_and_return_conditional_losses_18768Э
'random_rotation/StatefulPartitionedCallStatefulPartitionedCall,random_flip/StatefulPartitionedCall:output:0random_rotation_18890*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_random_rotation_layer_call_and_return_conditional_losses_18889Й
IdentityIdentity0random_rotation/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ААr
NoOpNoOp$^random_flip/StatefulPartitionedCall(^random_rotation/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         АА: : 2J
#random_flip/StatefulPartitionedCall#random_flip/StatefulPartitionedCall2R
'random_rotation/StatefulPartitionedCall'random_rotation/StatefulPartitionedCall:d `
1
_output_shapes
:         АА
+
_user_specified_namerandom_flip_input:%!

_user_specified_name18769:%!

_user_specified_name18890
╖
№
C__inference_conv2d_5_layer_call_and_return_conditional_losses_19015

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ~~ *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ~~ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         ~~ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         ~~ S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         АА: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         АА
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
═f
╨
 __inference__wrapped_model_18609
sequential_inputN
4sequential_3_conv2d_5_conv2d_readvariableop_resource: C
5sequential_3_conv2d_5_biasadd_readvariableop_resource: N
4sequential_3_conv2d_6_conv2d_readvariableop_resource: @C
5sequential_3_conv2d_6_biasadd_readvariableop_resource:@N
4sequential_3_conv2d_7_conv2d_readvariableop_resource:@@C
5sequential_3_conv2d_7_biasadd_readvariableop_resource:@N
4sequential_3_conv2d_8_conv2d_readvariableop_resource:@@C
5sequential_3_conv2d_8_biasadd_readvariableop_resource:@N
4sequential_3_conv2d_9_conv2d_readvariableop_resource:@@C
5sequential_3_conv2d_9_biasadd_readvariableop_resource:@F
3sequential_3_dense_2_matmul_readvariableop_resource:	А@B
4sequential_3_dense_2_biasadd_readvariableop_resource:@E
3sequential_3_dense_3_matmul_readvariableop_resource:@B
4sequential_3_dense_3_biasadd_readvariableop_resource:
identityИв,sequential_3/conv2d_5/BiasAdd/ReadVariableOpв+sequential_3/conv2d_5/Conv2D/ReadVariableOpв,sequential_3/conv2d_6/BiasAdd/ReadVariableOpв+sequential_3/conv2d_6/Conv2D/ReadVariableOpв,sequential_3/conv2d_7/BiasAdd/ReadVariableOpв+sequential_3/conv2d_7/Conv2D/ReadVariableOpв,sequential_3/conv2d_8/BiasAdd/ReadVariableOpв+sequential_3/conv2d_8/Conv2D/ReadVariableOpв,sequential_3/conv2d_9/BiasAdd/ReadVariableOpв+sequential_3/conv2d_9/Conv2D/ReadVariableOpв+sequential_3/dense_2/BiasAdd/ReadVariableOpв*sequential_3/dense_2/MatMul/ReadVariableOpв+sequential_3/dense_3/BiasAdd/ReadVariableOpв*sequential_3/dense_3/MatMul/ReadVariableOp}
,sequential_3/sequential/resizing/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"А   А   ч
6sequential_3/sequential/resizing/resize/ResizeBilinearResizeBilinearsequential_input5sequential_3/sequential/resizing/resize/size:output:0*
T0*1
_output_shapes
:         АА*
half_pixel_centers(m
(sequential_3/sequential/rescaling/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *БАА;o
*sequential_3/sequential/rescaling/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    ф
%sequential_3/sequential/rescaling/mulMulGsequential_3/sequential/resizing/resize/ResizeBilinear:resized_images:01sequential_3/sequential/rescaling/Cast/x:output:0*
T0*1
_output_shapes
:         АА╩
%sequential_3/sequential/rescaling/addAddV2)sequential_3/sequential/rescaling/mul:z:03sequential_3/sequential/rescaling/Cast_1/x:output:0*
T0*1
_output_shapes
:         ААи
+sequential_3/conv2d_5/Conv2D/ReadVariableOpReadVariableOp4sequential_3_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0щ
sequential_3/conv2d_5/Conv2DConv2D)sequential_3/sequential/rescaling/add:z:03sequential_3/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ~~ *
paddingVALID*
strides
Ю
,sequential_3/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0┐
sequential_3/conv2d_5/BiasAddBiasAdd%sequential_3/conv2d_5/Conv2D:output:04sequential_3/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ~~ Д
sequential_3/conv2d_5/ReluRelu&sequential_3/conv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:         ~~ ╞
$sequential_3/max_pooling2d_5/MaxPoolMaxPool(sequential_3/conv2d_5/Relu:activations:0*/
_output_shapes
:         ?? *
ksize
*
paddingVALID*
strides
и
+sequential_3/conv2d_6/Conv2D/ReadVariableOpReadVariableOp4sequential_3_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0э
sequential_3/conv2d_6/Conv2DConv2D-sequential_3/max_pooling2d_5/MaxPool:output:03sequential_3/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ==@*
paddingVALID*
strides
Ю
,sequential_3/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0┐
sequential_3/conv2d_6/BiasAddBiasAdd%sequential_3/conv2d_6/Conv2D:output:04sequential_3/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ==@Д
sequential_3/conv2d_6/ReluRelu&sequential_3/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:         ==@╞
$sequential_3/max_pooling2d_6/MaxPoolMaxPool(sequential_3/conv2d_6/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
и
+sequential_3/conv2d_7/Conv2D/ReadVariableOpReadVariableOp4sequential_3_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0э
sequential_3/conv2d_7/Conv2DConv2D-sequential_3/max_pooling2d_6/MaxPool:output:03sequential_3/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
Ю
,sequential_3/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0┐
sequential_3/conv2d_7/BiasAddBiasAdd%sequential_3/conv2d_7/Conv2D:output:04sequential_3/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @Д
sequential_3/conv2d_7/ReluRelu&sequential_3/conv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:         @╞
$sequential_3/max_pooling2d_7/MaxPoolMaxPool(sequential_3/conv2d_7/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
и
+sequential_3/conv2d_8/Conv2D/ReadVariableOpReadVariableOp4sequential_3_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0э
sequential_3/conv2d_8/Conv2DConv2D-sequential_3/max_pooling2d_7/MaxPool:output:03sequential_3/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
Ю
,sequential_3/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0┐
sequential_3/conv2d_8/BiasAddBiasAdd%sequential_3/conv2d_8/Conv2D:output:04sequential_3/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @Д
sequential_3/conv2d_8/ReluRelu&sequential_3/conv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:         @╞
$sequential_3/max_pooling2d_8/MaxPoolMaxPool(sequential_3/conv2d_8/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
и
+sequential_3/conv2d_9/Conv2D/ReadVariableOpReadVariableOp4sequential_3_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0э
sequential_3/conv2d_9/Conv2DConv2D-sequential_3/max_pooling2d_8/MaxPool:output:03sequential_3/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
Ю
,sequential_3/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0┐
sequential_3/conv2d_9/BiasAddBiasAdd%sequential_3/conv2d_9/Conv2D:output:04sequential_3/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @Д
sequential_3/conv2d_9/ReluRelu&sequential_3/conv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:         @╞
$sequential_3/max_pooling2d_9/MaxPoolMaxPool(sequential_3/conv2d_9/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
m
sequential_3/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ▓
sequential_3/flatten_1/ReshapeReshape-sequential_3/max_pooling2d_9/MaxPool:output:0%sequential_3/flatten_1/Const:output:0*
T0*(
_output_shapes
:         АЯ
*sequential_3/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_2_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype0┤
sequential_3/dense_2/MatMulMatMul'sequential_3/flatten_1/Reshape:output:02sequential_3/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ь
+sequential_3/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╡
sequential_3/dense_2/BiasAddBiasAdd%sequential_3/dense_2/MatMul:product:03sequential_3/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @z
sequential_3/dense_2/ReluRelu%sequential_3/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:         @Ю
*sequential_3/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_3_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0┤
sequential_3/dense_3/MatMulMatMul'sequential_3/dense_2/Relu:activations:02sequential_3/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ь
+sequential_3/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╡
sequential_3/dense_3/BiasAddBiasAdd%sequential_3/dense_3/MatMul:product:03sequential_3/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         А
sequential_3/dense_3/SoftmaxSoftmax%sequential_3/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:         u
IdentityIdentity&sequential_3/dense_3/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         й
NoOpNoOp-^sequential_3/conv2d_5/BiasAdd/ReadVariableOp,^sequential_3/conv2d_5/Conv2D/ReadVariableOp-^sequential_3/conv2d_6/BiasAdd/ReadVariableOp,^sequential_3/conv2d_6/Conv2D/ReadVariableOp-^sequential_3/conv2d_7/BiasAdd/ReadVariableOp,^sequential_3/conv2d_7/Conv2D/ReadVariableOp-^sequential_3/conv2d_8/BiasAdd/ReadVariableOp,^sequential_3/conv2d_8/Conv2D/ReadVariableOp-^sequential_3/conv2d_9/BiasAdd/ReadVariableOp,^sequential_3/conv2d_9/Conv2D/ReadVariableOp,^sequential_3/dense_2/BiasAdd/ReadVariableOp+^sequential_3/dense_2/MatMul/ReadVariableOp,^sequential_3/dense_3/BiasAdd/ReadVariableOp+^sequential_3/dense_3/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:         АА: : : : : : : : : : : : : : 2\
,sequential_3/conv2d_5/BiasAdd/ReadVariableOp,sequential_3/conv2d_5/BiasAdd/ReadVariableOp2Z
+sequential_3/conv2d_5/Conv2D/ReadVariableOp+sequential_3/conv2d_5/Conv2D/ReadVariableOp2\
,sequential_3/conv2d_6/BiasAdd/ReadVariableOp,sequential_3/conv2d_6/BiasAdd/ReadVariableOp2Z
+sequential_3/conv2d_6/Conv2D/ReadVariableOp+sequential_3/conv2d_6/Conv2D/ReadVariableOp2\
,sequential_3/conv2d_7/BiasAdd/ReadVariableOp,sequential_3/conv2d_7/BiasAdd/ReadVariableOp2Z
+sequential_3/conv2d_7/Conv2D/ReadVariableOp+sequential_3/conv2d_7/Conv2D/ReadVariableOp2\
,sequential_3/conv2d_8/BiasAdd/ReadVariableOp,sequential_3/conv2d_8/BiasAdd/ReadVariableOp2Z
+sequential_3/conv2d_8/Conv2D/ReadVariableOp+sequential_3/conv2d_8/Conv2D/ReadVariableOp2\
,sequential_3/conv2d_9/BiasAdd/ReadVariableOp,sequential_3/conv2d_9/BiasAdd/ReadVariableOp2Z
+sequential_3/conv2d_9/Conv2D/ReadVariableOp+sequential_3/conv2d_9/Conv2D/ReadVariableOp2Z
+sequential_3/dense_2/BiasAdd/ReadVariableOp+sequential_3/dense_2/BiasAdd/ReadVariableOp2X
*sequential_3/dense_2/MatMul/ReadVariableOp*sequential_3/dense_2/MatMul/ReadVariableOp2Z
+sequential_3/dense_3/BiasAdd/ReadVariableOp+sequential_3/dense_3/BiasAdd/ReadVariableOp2X
*sequential_3/dense_3/MatMul/ReadVariableOp*sequential_3/dense_3/MatMul/ReadVariableOp:c _
1
_output_shapes
:         АА
*
_user_specified_namesequential_input:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
╬@
╚
G__inference_sequential_3_layer_call_and_return_conditional_losses_19177
sequential_input(
conv2d_5_19135: 
conv2d_5_19137: (
conv2d_6_19141: @
conv2d_6_19143:@(
conv2d_7_19147:@@
conv2d_7_19149:@(
conv2d_8_19153:@@
conv2d_8_19155:@(
conv2d_9_19159:@@
conv2d_9_19161:@ 
dense_2_19166:	А@
dense_2_19168:@
dense_3_19171:@
dense_3_19173:
identityИв conv2d_5/StatefulPartitionedCallв conv2d_6/StatefulPartitionedCallв conv2d_7/StatefulPartitionedCallв conv2d_8/StatefulPartitionedCallв conv2d_9/StatefulPartitionedCallвdense_2/StatefulPartitionedCallвdense_3/StatefulPartitionedCall╧
sequential/PartitionedCallPartitionedCallsequential_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_18635ц
sequential_1/PartitionedCallPartitionedCall#sequential/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_18908Ф
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall%sequential_1/PartitionedCall:output:0conv2d_5_19135conv2d_5_19137*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ~~ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_19015Ё
max_pooling2d_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ?? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_18951Ч
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0conv2d_6_19141conv2d_6_19143*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ==@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_19032Ё
max_pooling2d_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_18961Ч
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0conv2d_7_19147conv2d_7_19149*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_19049Ё
max_pooling2d_7/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_18971Ч
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_7/PartitionedCall:output:0conv2d_8_19153conv2d_8_19155*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_19066Ё
max_pooling2d_8/PartitionedCallPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_18981Ч
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_8/PartitionedCall:output:0conv2d_9_19159conv2d_9_19161*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_19083Ё
max_pooling2d_9/PartitionedCallPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_18991▄
flatten_1/PartitionedCallPartitionedCall(max_pooling2d_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_19095Е
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_19166dense_2_19168*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_19107Л
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_19171dense_3_19173*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_19123w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Х
NoOpNoOp!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:         АА: : : : : : : : : : : : : : 2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:c _
1
_output_shapes
:         АА
*
_user_specified_namesequential_input:%!

_user_specified_name19135:%!

_user_specified_name19137:%!

_user_specified_name19141:%!

_user_specified_name19143:%!

_user_specified_name19147:%!

_user_specified_name19149:%!

_user_specified_name19153:%!

_user_specified_name19155:%	!

_user_specified_name19159:%
!

_user_specified_name19161:%!

_user_specified_name19166:%!

_user_specified_name19168:%!

_user_specified_name19171:%!

_user_specified_name19173
│
№
C__inference_conv2d_8_layer_call_and_return_conditional_losses_19066

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Т
f
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_18981

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
│
№
C__inference_conv2d_7_layer_call_and_return_conditional_losses_19049

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
п
E
)__inference_flatten_1_layer_call_fn_19510

inputs
identity░
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_19095a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
┼А
├
J__inference_random_rotation_layer_call_and_return_conditional_losses_19837

inputs6
(stateful_uniform_rngreadandskip_resource:	
identityИвstateful_uniform/RngReadAndSkipI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
¤        j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
■        a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: d
stateful_uniform/shapePackstrided_slice:output:0*
N*
T0*
_output_shapes
:Y
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|┘а┐Y
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|┘а?`
stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: А
stateful_uniform/ProdProdstateful_uniform/shape:output:0stateful_uniform/Const:output:0*
T0*
_output_shapes
: Y
stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :o
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ╢
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:n
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:м
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask}
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0p
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:в
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:Б
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0o
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :Л
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:         z
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: Ч
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*#
_output_shapes
:         А
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*#
_output_shapes
:         Z
rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?g
rotation_matrix/subSub
Cast_1:y:0rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: ^
rotation_matrix/CosCosstateful_uniform:z:0*
T0*#
_output_shapes
:         \
rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?k
rotation_matrix/sub_1Sub
Cast_1:y:0 rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: |
rotation_matrix/mulMulrotation_matrix/Cos:y:0rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:         ^
rotation_matrix/SinSinstateful_uniform:z:0*
T0*#
_output_shapes
:         \
rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?i
rotation_matrix/sub_2SubCast:y:0 rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: ~
rotation_matrix/mul_1Mulrotation_matrix/Sin:y:0rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:         ~
rotation_matrix/sub_3Subrotation_matrix/mul:z:0rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:         ~
rotation_matrix/sub_4Subrotation_matrix/sub:z:0rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:         ^
rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @П
rotation_matrix/truedivRealDivrotation_matrix/sub_4:z:0"rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:         \
rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?i
rotation_matrix/sub_5SubCast:y:0 rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: `
rotation_matrix/Sin_1Sinstateful_uniform:z:0*
T0*#
_output_shapes
:         \
rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?k
rotation_matrix/sub_6Sub
Cast_1:y:0 rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: А
rotation_matrix/mul_2Mulrotation_matrix/Sin_1:y:0rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:         `
rotation_matrix/Cos_1Cosstateful_uniform:z:0*
T0*#
_output_shapes
:         \
rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?i
rotation_matrix/sub_7SubCast:y:0 rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: А
rotation_matrix/mul_3Mulrotation_matrix/Cos_1:y:0rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:         А
rotation_matrix/addAddV2rotation_matrix/mul_2:z:0rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:         ~
rotation_matrix/sub_8Subrotation_matrix/sub_5:z:0rotation_matrix/add:z:0*
T0*#
_output_shapes
:         `
rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @У
rotation_matrix/truediv_1RealDivrotation_matrix/sub_8:z:0$rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:         g
rotation_matrix/ShapeShapestateful_uniform:z:0*
T0*
_output_shapes
::э╧m
#rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
rotation_matrix/strided_sliceStridedSlicerotation_matrix/Shape:output:0,rotation_matrix/strided_slice/stack:output:0.rotation_matrix/strided_slice/stack_1:output:0.rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
rotation_matrix/Cos_2Cosstateful_uniform:z:0*
T0*#
_output_shapes
:         v
%rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
rotation_matrix/strided_slice_1StridedSlicerotation_matrix/Cos_2:y:0.rotation_matrix/strided_slice_1/stack:output:00rotation_matrix/strided_slice_1/stack_1:output:00rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/Sin_2Sinstateful_uniform:z:0*
T0*#
_output_shapes
:         v
%rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
rotation_matrix/strided_slice_2StridedSlicerotation_matrix/Sin_2:y:0.rotation_matrix/strided_slice_2/stack:output:00rotation_matrix/strided_slice_2/stack_1:output:00rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_maskv
rotation_matrix/NegNeg(rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:         v
%rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╓
rotation_matrix/strided_slice_3StridedSlicerotation_matrix/truediv:z:0.rotation_matrix/strided_slice_3/stack:output:00rotation_matrix/strided_slice_3/stack_1:output:00rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/Sin_3Sinstateful_uniform:z:0*
T0*#
_output_shapes
:         v
%rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
rotation_matrix/strided_slice_4StridedSlicerotation_matrix/Sin_3:y:0.rotation_matrix/strided_slice_4/stack:output:00rotation_matrix/strided_slice_4/stack_1:output:00rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/Cos_3Cosstateful_uniform:z:0*
T0*#
_output_shapes
:         v
%rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
rotation_matrix/strided_slice_5StridedSlicerotation_matrix/Cos_3:y:0.rotation_matrix/strided_slice_5/stack:output:00rotation_matrix/strided_slice_5/stack_1:output:00rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_maskv
%rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╪
rotation_matrix/strided_slice_6StridedSlicerotation_matrix/truediv_1:z:0.rotation_matrix/strided_slice_6/stack:output:00rotation_matrix/strided_slice_6/stack_1:output:00rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :г
rotation_matrix/zeros/packedPack&rotation_matrix/strided_slice:output:0'rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:`
rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ь
rotation_matrix/zerosFill%rotation_matrix/zeros/packed:output:0$rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:         ]
rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :О
rotation_matrix/concatConcatV2(rotation_matrix/strided_slice_1:output:0rotation_matrix/Neg:y:0(rotation_matrix/strided_slice_3:output:0(rotation_matrix/strided_slice_4:output:0(rotation_matrix/strided_slice_5:output:0(rotation_matrix/strided_slice_6:output:0rotation_matrix/zeros:output:0$rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:         S
transform/ShapeShapeinputs*
T0*
_output_shapes
::э╧g
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:я
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:Y
transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    б
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputsrotation_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*1
_output_shapes
:         АА*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEARТ
IdentityIdentity9transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:         ААD
NoOpNoOp ^stateful_uniform/RngReadAndSkip*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         АА: 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:         АА
 
_user_specified_nameinputs:($
"
_user_specified_name
resource
гЧ
■
F__inference_random_flip_layer_call_and_return_conditional_losses_18768

inputs?
1stateful_uniform_full_int_rngreadandskip_resource:	
identityИв(stateful_uniform_full_int/RngReadAndSkipв*stateful_uniform_full_int_1/RngReadAndSkipi
stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:i
stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ы
stateful_uniform_full_int/ProdProd(stateful_uniform_full_int/shape:output:0(stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: b
 stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :Б
 stateful_uniform_full_int/Cast_1Cast'stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ┌
(stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip1stateful_uniform_full_int_rngreadandskip_resource)stateful_uniform_full_int/Cast/x:output:0$stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:w
-stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
'stateful_uniform_full_int/strided_sliceStridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:06stateful_uniform_full_int/strided_slice/stack:output:08stateful_uniform_full_int/strided_slice/stack_1:output:08stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_maskП
!stateful_uniform_full_int/BitcastBitcast0stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0y
/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╧
)stateful_uniform_full_int/strided_slice_1StridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:08stateful_uniform_full_int/strided_slice_1/stack:output:0:stateful_uniform_full_int/strided_slice_1/stack_1:output:0:stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:У
#stateful_uniform_full_int/Bitcast_1Bitcast2stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0_
stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :С
stateful_uniform_full_intStatelessRandomUniformFullIntV2(stateful_uniform_full_int/shape:output:0,stateful_uniform_full_int/Bitcast_1:output:0*stateful_uniform_full_int/Bitcast:output:0&stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	T

zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R x
stackPack"stateful_uniform_full_int:output:0zeros_like:output:0*
N*
T0	*
_output_shapes

:d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ў
strided_sliceStridedSlicestack:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_maskЮ
3stateless_random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*1
_output_shapes
:         ААа
&stateless_random_flip_left_right/ShapeShape<stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
::э╧~
4stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: А
6stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:А
6stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ў
.stateless_random_flip_left_right/strided_sliceStridedSlice/stateless_random_flip_left_right/Shape:output:0=stateless_random_flip_left_right/strided_slice/stack:output:0?stateless_random_flip_left_right/strided_slice/stack_1:output:0?stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskо
?stateless_random_flip_left_right/stateless_random_uniform/shapePack7stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:В
=stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    В
=stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  А?░
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice:output:0* 
_output_shapes
::Ш
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :■
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Hstateless_random_flip_left_right/stateless_random_uniform/shape:output:0\stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0`stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0_stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:         ї
=stateless_random_flip_left_right/stateless_random_uniform/subSubFstateless_random_flip_left_right/stateless_random_uniform/max:output:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: Т
=stateless_random_flip_left_right/stateless_random_uniform/mulMul[stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Astateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:         √
9stateless_random_flip_left_right/stateless_random_uniformAddV2Astateless_random_flip_left_right/stateless_random_uniform/mul:z:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:         r
0stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :r
0stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :r
0stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :╬
.stateless_random_flip_left_right/Reshape/shapePack7stateless_random_flip_left_right/strided_slice:output:09stateless_random_flip_left_right/Reshape/shape/1:output:09stateless_random_flip_left_right/Reshape/shape/2:output:09stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:х
(stateless_random_flip_left_right/ReshapeReshape=stateless_random_flip_left_right/stateless_random_uniform:z:07stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:         Ь
&stateless_random_flip_left_right/RoundRound1stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:         y
/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:ы
*stateless_random_flip_left_right/ReverseV2	ReverseV2<stateless_random_flip_left_right/control_dependency:output:08stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:         АА╚
$stateless_random_flip_left_right/mulMul*stateless_random_flip_left_right/Round:y:03stateless_random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:         ААk
&stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?┬
$stateless_random_flip_left_right/subSub/stateless_random_flip_left_right/sub/x:output:0*stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:         ╤
&stateless_random_flip_left_right/mul_1Mul(stateless_random_flip_left_right/sub:z:0<stateless_random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:         АА┐
$stateless_random_flip_left_right/addAddV2(stateless_random_flip_left_right/mul:z:0*stateless_random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:         ААk
!stateful_uniform_full_int_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:k
!stateful_uniform_full_int_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: б
 stateful_uniform_full_int_1/ProdProd*stateful_uniform_full_int_1/shape:output:0*stateful_uniform_full_int_1/Const:output:0*
T0*
_output_shapes
: d
"stateful_uniform_full_int_1/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :Е
"stateful_uniform_full_int_1/Cast_1Cast)stateful_uniform_full_int_1/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: Л
*stateful_uniform_full_int_1/RngReadAndSkipRngReadAndSkip1stateful_uniform_full_int_rngreadandskip_resource+stateful_uniform_full_int_1/Cast/x:output:0&stateful_uniform_full_int_1/Cast_1:y:0)^stateful_uniform_full_int/RngReadAndSkip*
_output_shapes
:y
/stateful_uniform_full_int_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1stateful_uniform_full_int_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:у
)stateful_uniform_full_int_1/strided_sliceStridedSlice2stateful_uniform_full_int_1/RngReadAndSkip:value:08stateful_uniform_full_int_1/strided_slice/stack:output:0:stateful_uniform_full_int_1/strided_slice/stack_1:output:0:stateful_uniform_full_int_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_maskУ
#stateful_uniform_full_int_1/BitcastBitcast2stateful_uniform_full_int_1/strided_slice:output:0*
T0	*
_output_shapes
:*

type0{
1stateful_uniform_full_int_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3stateful_uniform_full_int_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3stateful_uniform_full_int_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
+stateful_uniform_full_int_1/strided_slice_1StridedSlice2stateful_uniform_full_int_1/RngReadAndSkip:value:0:stateful_uniform_full_int_1/strided_slice_1/stack:output:0<stateful_uniform_full_int_1/strided_slice_1/stack_1:output:0<stateful_uniform_full_int_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:Ч
%stateful_uniform_full_int_1/Bitcast_1Bitcast4stateful_uniform_full_int_1/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0a
stateful_uniform_full_int_1/algConst*
_output_shapes
: *
dtype0*
value	B :Ы
stateful_uniform_full_int_1StatelessRandomUniformFullIntV2*stateful_uniform_full_int_1/shape:output:0.stateful_uniform_full_int_1/Bitcast_1:output:0,stateful_uniform_full_int_1/Bitcast:output:0(stateful_uniform_full_int_1/alg:output:0*
_output_shapes
:*
dtype0	V
zeros_like_1Const*
_output_shapes
:*
dtype0	*
valueB	R ~
stack_1Pack$stateful_uniform_full_int_1:output:0zeros_like_1:output:0*
N*
T0	*
_output_shapes

:f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Б
strided_slice_1StridedSlicestack_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask█
0stateless_random_flip_up_down/control_dependencyIdentity(stateless_random_flip_left_right/add:z:0*
T0*7
_class-
+)loc:@stateless_random_flip_left_right/add*1
_output_shapes
:         ААЪ
#stateless_random_flip_up_down/ShapeShape9stateless_random_flip_up_down/control_dependency:output:0*
T0*
_output_shapes
::э╧{
1stateless_random_flip_up_down/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3stateless_random_flip_up_down/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3stateless_random_flip_up_down/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
+stateless_random_flip_up_down/strided_sliceStridedSlice,stateless_random_flip_up_down/Shape:output:0:stateless_random_flip_up_down/strided_slice/stack:output:0<stateless_random_flip_up_down/strided_slice/stack_1:output:0<stateless_random_flip_up_down/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskи
<stateless_random_flip_up_down/stateless_random_uniform/shapePack4stateless_random_flip_up_down/strided_slice:output:0*
N*
T0*
_output_shapes
:
:stateless_random_flip_up_down/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    
:stateless_random_flip_up_down/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  А?п
Sstateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice_1:output:0* 
_output_shapes
::Х
Sstateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :я
Ostateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Estateless_random_flip_up_down/stateless_random_uniform/shape:output:0Ystateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0]stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0\stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:         ь
:stateless_random_flip_up_down/stateless_random_uniform/subSubCstateless_random_flip_up_down/stateless_random_uniform/max:output:0Cstateless_random_flip_up_down/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: Й
:stateless_random_flip_up_down/stateless_random_uniform/mulMulXstateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2:output:0>stateless_random_flip_up_down/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:         Є
6stateless_random_flip_up_down/stateless_random_uniformAddV2>stateless_random_flip_up_down/stateless_random_uniform/mul:z:0Cstateless_random_flip_up_down/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:         o
-stateless_random_flip_up_down/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :o
-stateless_random_flip_up_down/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :o
-stateless_random_flip_up_down/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :┐
+stateless_random_flip_up_down/Reshape/shapePack4stateless_random_flip_up_down/strided_slice:output:06stateless_random_flip_up_down/Reshape/shape/1:output:06stateless_random_flip_up_down/Reshape/shape/2:output:06stateless_random_flip_up_down/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:▄
%stateless_random_flip_up_down/ReshapeReshape:stateless_random_flip_up_down/stateless_random_uniform:z:04stateless_random_flip_up_down/Reshape/shape:output:0*
T0*/
_output_shapes
:         Ц
#stateless_random_flip_up_down/RoundRound.stateless_random_flip_up_down/Reshape:output:0*
T0*/
_output_shapes
:         v
,stateless_random_flip_up_down/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:т
'stateless_random_flip_up_down/ReverseV2	ReverseV29stateless_random_flip_up_down/control_dependency:output:05stateless_random_flip_up_down/ReverseV2/axis:output:0*
T0*1
_output_shapes
:         АА┐
!stateless_random_flip_up_down/mulMul'stateless_random_flip_up_down/Round:y:00stateless_random_flip_up_down/ReverseV2:output:0*
T0*1
_output_shapes
:         ААh
#stateless_random_flip_up_down/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╣
!stateless_random_flip_up_down/subSub,stateless_random_flip_up_down/sub/x:output:0'stateless_random_flip_up_down/Round:y:0*
T0*/
_output_shapes
:         ╚
#stateless_random_flip_up_down/mul_1Mul%stateless_random_flip_up_down/sub:z:09stateless_random_flip_up_down/control_dependency:output:0*
T0*1
_output_shapes
:         АА╢
!stateless_random_flip_up_down/addAddV2%stateless_random_flip_up_down/mul:z:0'stateless_random_flip_up_down/mul_1:z:0*
T0*1
_output_shapes
:         АА~
IdentityIdentity%stateless_random_flip_up_down/add:z:0^NoOp*
T0*1
_output_shapes
:         ААz
NoOpNoOp)^stateful_uniform_full_int/RngReadAndSkip+^stateful_uniform_full_int_1/RngReadAndSkip*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         АА: 2T
(stateful_uniform_full_int/RngReadAndSkip(stateful_uniform_full_int/RngReadAndSkip2X
*stateful_uniform_full_int_1/RngReadAndSkip*stateful_uniform_full_int_1/RngReadAndSkip:Y U
1
_output_shapes
:         АА
 
_user_specified_nameinputs:($
"
_user_specified_name
resource
│
№
C__inference_conv2d_6_layer_call_and_return_conditional_losses_19032

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ==@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ==@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         ==@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         ==@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         ?? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         ?? 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
═

Ї
B__inference_dense_2_layer_call_and_return_conditional_losses_19536

inputs1
matmul_readvariableop_resource:	А@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         @a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         @S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
═

Ї
B__inference_dense_2_layer_call_and_return_conditional_losses_19107

inputs1
matmul_readvariableop_resource:	А@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         @a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         @S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ф
b
F__inference_random_flip_layer_call_and_return_conditional_losses_19707

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:         АА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         АА:Y U
1
_output_shapes
:         АА
 
_user_specified_nameinputs
Ў
`
D__inference_rescaling_layer_call_and_return_conditional_losses_19580

inputs
identityK
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *БАА;M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    _
mulMulinputsCast/x:output:0*
T0*1
_output_shapes
:         ААd
addAddV2mul:z:0Cast_1/x:output:0*
T0*1
_output_shapes
:         ААY
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:         АА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         АА:Y U
1
_output_shapes
:         АА
 
_user_specified_nameinputs
ы
Х
'__inference_dense_2_layer_call_fn_19525

inputs
unknown:	А@
	unknown_0:@
identityИвStatefulPartitionedCall╫
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_19107o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs:%!

_user_specified_name19519:%!

_user_specified_name19521
╢
K
/__inference_max_pooling2d_9_layer_call_fn_19500

inputs
identity╪
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_18991Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Т
f
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_19475

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
│
№
C__inference_conv2d_8_layer_call_and_return_conditional_losses_19465

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
╢
K
/__inference_max_pooling2d_5_layer_call_fn_19380

inputs
identity╪
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_18951Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
▀
N
*__inference_sequential_layer_call_fn_18640
resizing_input
identity┬
PartitionedCallPartitionedCallresizing_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_18629j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         АА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         АА:a ]
1
_output_shapes
:         АА
(
_user_specified_nameresizing_input
Я
У
,__inference_sequential_3_layer_call_fn_19247
sequential_input!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@#
	unknown_7:@@
	unknown_8:@
	unknown_9:	А@

unknown_10:@

unknown_11:@

unknown_12:
identityИвStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallsequential_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_19177o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:         АА: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
1
_output_shapes
:         АА
*
_user_specified_namesequential_input:%!

_user_specified_name19217:%!

_user_specified_name19219:%!

_user_specified_name19221:%!

_user_specified_name19223:%!

_user_specified_name19225:%!

_user_specified_name19227:%!

_user_specified_name19229:%!

_user_specified_name19231:%	!

_user_specified_name19233:%
!

_user_specified_name19235:%!

_user_specified_name19237:%!

_user_specified_name19239:%!

_user_specified_name19241:%!

_user_specified_name19243
░∙
р 
!__inference__traced_restore_20359
file_prefix:
 assignvariableop_conv2d_5_kernel: .
 assignvariableop_1_conv2d_5_bias: <
"assignvariableop_2_conv2d_6_kernel: @.
 assignvariableop_3_conv2d_6_bias:@<
"assignvariableop_4_conv2d_7_kernel:@@.
 assignvariableop_5_conv2d_7_bias:@<
"assignvariableop_6_conv2d_8_kernel:@@.
 assignvariableop_7_conv2d_8_bias:@<
"assignvariableop_8_conv2d_9_kernel:@@.
 assignvariableop_9_conv2d_9_bias:@5
"assignvariableop_10_dense_2_kernel:	А@.
 assignvariableop_11_dense_2_bias:@4
"assignvariableop_12_dense_3_kernel:@.
 assignvariableop_13_dense_3_bias:'
assignvariableop_14_adam_iter:	 )
assignvariableop_15_adam_beta_1: )
assignvariableop_16_adam_beta_2: (
assignvariableop_17_adam_decay: 0
&assignvariableop_18_adam_learning_rate: %
assignvariableop_19_total_1: %
assignvariableop_20_count_1: #
assignvariableop_21_total: #
assignvariableop_22_count: 6
(assignvariableop_23_random_flip_statevar:	:
,assignvariableop_24_random_rotation_statevar:	D
*assignvariableop_25_adam_conv2d_5_kernel_m: 6
(assignvariableop_26_adam_conv2d_5_bias_m: D
*assignvariableop_27_adam_conv2d_6_kernel_m: @6
(assignvariableop_28_adam_conv2d_6_bias_m:@D
*assignvariableop_29_adam_conv2d_7_kernel_m:@@6
(assignvariableop_30_adam_conv2d_7_bias_m:@D
*assignvariableop_31_adam_conv2d_8_kernel_m:@@6
(assignvariableop_32_adam_conv2d_8_bias_m:@D
*assignvariableop_33_adam_conv2d_9_kernel_m:@@6
(assignvariableop_34_adam_conv2d_9_bias_m:@<
)assignvariableop_35_adam_dense_2_kernel_m:	А@5
'assignvariableop_36_adam_dense_2_bias_m:@;
)assignvariableop_37_adam_dense_3_kernel_m:@5
'assignvariableop_38_adam_dense_3_bias_m:D
*assignvariableop_39_adam_conv2d_5_kernel_v: 6
(assignvariableop_40_adam_conv2d_5_bias_v: D
*assignvariableop_41_adam_conv2d_6_kernel_v: @6
(assignvariableop_42_adam_conv2d_6_bias_v:@D
*assignvariableop_43_adam_conv2d_7_kernel_v:@@6
(assignvariableop_44_adam_conv2d_7_bias_v:@D
*assignvariableop_45_adam_conv2d_8_kernel_v:@@6
(assignvariableop_46_adam_conv2d_8_bias_v:@D
*assignvariableop_47_adam_conv2d_9_kernel_v:@@6
(assignvariableop_48_adam_conv2d_9_bias_v:@<
)assignvariableop_49_adam_dense_2_kernel_v:	А@5
'assignvariableop_50_adam_dense_2_bias_v:@;
)assignvariableop_51_adam_dense_3_kernel_v:@5
'assignvariableop_52_adam_dense_3_bias_v:
identity_54ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_45вAssignVariableOp_46вAssignVariableOp_47вAssignVariableOp_48вAssignVariableOp_49вAssignVariableOp_5вAssignVariableOp_50вAssignVariableOp_51вAssignVariableOp_52вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9И
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*о
valueдBб6B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer-1/layer-0/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBRlayer-1/layer-1/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH▄
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*
valuevBt6B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B п
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ю
_output_shapes█
╪::::::::::::::::::::::::::::::::::::::::::::::::::::::*D
dtypes:
826			[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:│
AssignVariableOpAssignVariableOp assignvariableop_conv2d_5_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_5_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_6_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_6_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_7_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_7_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_8_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_8_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_8AssignVariableOp"assignvariableop_8_conv2d_9_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv2d_9_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_2_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_2_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_3_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_3_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:╢
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_iterIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:╕
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_beta_1Identity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:╕
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_beta_2Identity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_decayIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_learning_rateIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_1Identity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_1Identity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_21AssignVariableOpassignvariableop_21_totalIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_22AssignVariableOpassignvariableop_22_countIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0	*
_output_shapes
:┴
AssignVariableOp_23AssignVariableOp(assignvariableop_23_random_flip_statevarIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0	*
_output_shapes
:┼
AssignVariableOp_24AssignVariableOp,assignvariableop_24_random_rotation_statevarIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_conv2d_5_kernel_mIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_conv2d_5_bias_mIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_conv2d_6_kernel_mIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_conv2d_6_bias_mIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_conv2d_7_kernel_mIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_conv2d_7_bias_mIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_conv2d_8_kernel_mIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_conv2d_8_bias_mIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_conv2d_9_kernel_mIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_conv2d_9_bias_mIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_dense_2_kernel_mIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adam_dense_2_bias_mIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_37AssignVariableOp)assignvariableop_37_adam_dense_3_kernel_mIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_38AssignVariableOp'assignvariableop_38_adam_dense_3_bias_mIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_conv2d_5_kernel_vIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_conv2d_5_bias_vIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_conv2d_6_kernel_vIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_conv2d_6_bias_vIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_conv2d_7_kernel_vIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_conv2d_7_bias_vIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_conv2d_8_kernel_vIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_conv2d_8_bias_vIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_conv2d_9_kernel_vIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_conv2d_9_bias_vIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_49AssignVariableOp)assignvariableop_49_adam_dense_2_kernel_vIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_50AssignVariableOp'assignvariableop_50_adam_dense_2_bias_vIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_51AssignVariableOp)assignvariableop_51_adam_dense_3_kernel_vIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_52AssignVariableOp'assignvariableop_52_adam_dense_3_bias_vIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 ▌	
Identity_53Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_54IdentityIdentity_53:output:0^NoOp_1*
T0*
_output_shapes
: ж	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_54Identity_54:output:0*(
_construction_contextkEagerRuntime*
_input_shapesn
l: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:/+
)
_user_specified_nameconv2d_5/kernel:-)
'
_user_specified_nameconv2d_5/bias:/+
)
_user_specified_nameconv2d_6/kernel:-)
'
_user_specified_nameconv2d_6/bias:/+
)
_user_specified_nameconv2d_7/kernel:-)
'
_user_specified_nameconv2d_7/bias:/+
)
_user_specified_nameconv2d_8/kernel:-)
'
_user_specified_nameconv2d_8/bias:/	+
)
_user_specified_nameconv2d_9/kernel:-
)
'
_user_specified_nameconv2d_9/bias:.*
(
_user_specified_namedense_2/kernel:,(
&
_user_specified_namedense_2/bias:.*
(
_user_specified_namedense_3/kernel:,(
&
_user_specified_namedense_3/bias:)%
#
_user_specified_name	Adam/iter:+'
%
_user_specified_nameAdam/beta_1:+'
%
_user_specified_nameAdam/beta_2:*&
$
_user_specified_name
Adam/decay:2.
,
_user_specified_nameAdam/learning_rate:'#
!
_user_specified_name	total_1:'#
!
_user_specified_name	count_1:%!

_user_specified_nametotal:%!

_user_specified_namecount:40
.
_user_specified_namerandom_flip/StateVar:84
2
_user_specified_namerandom_rotation/StateVar:62
0
_user_specified_nameAdam/conv2d_5/kernel/m:40
.
_user_specified_nameAdam/conv2d_5/bias/m:62
0
_user_specified_nameAdam/conv2d_6/kernel/m:40
.
_user_specified_nameAdam/conv2d_6/bias/m:62
0
_user_specified_nameAdam/conv2d_7/kernel/m:40
.
_user_specified_nameAdam/conv2d_7/bias/m:6 2
0
_user_specified_nameAdam/conv2d_8/kernel/m:4!0
.
_user_specified_nameAdam/conv2d_8/bias/m:6"2
0
_user_specified_nameAdam/conv2d_9/kernel/m:4#0
.
_user_specified_nameAdam/conv2d_9/bias/m:5$1
/
_user_specified_nameAdam/dense_2/kernel/m:3%/
-
_user_specified_nameAdam/dense_2/bias/m:5&1
/
_user_specified_nameAdam/dense_3/kernel/m:3'/
-
_user_specified_nameAdam/dense_3/bias/m:6(2
0
_user_specified_nameAdam/conv2d_5/kernel/v:4)0
.
_user_specified_nameAdam/conv2d_5/bias/v:6*2
0
_user_specified_nameAdam/conv2d_6/kernel/v:4+0
.
_user_specified_nameAdam/conv2d_6/bias/v:6,2
0
_user_specified_nameAdam/conv2d_7/kernel/v:4-0
.
_user_specified_nameAdam/conv2d_7/bias/v:6.2
0
_user_specified_nameAdam/conv2d_8/kernel/v:4/0
.
_user_specified_nameAdam/conv2d_8/bias/v:602
0
_user_specified_nameAdam/conv2d_9/kernel/v:410
.
_user_specified_nameAdam/conv2d_9/bias/v:521
/
_user_specified_nameAdam/dense_2/kernel/v:33/
-
_user_specified_nameAdam/dense_2/bias/v:541
/
_user_specified_nameAdam/dense_3/kernel/v:35/
-
_user_specified_nameAdam/dense_3/bias/v
┐
╟
,__inference_sequential_3_layer_call_fn_19214
sequential_input
unknown:	
	unknown_0:	#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@#
	unknown_7:@@
	unknown_8:@#
	unknown_9:@@

unknown_10:@

unknown_11:	А@

unknown_12:@

unknown_13:@

unknown_14:
identityИвStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallsequential_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_19130o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:         АА: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
1
_output_shapes
:         АА
*
_user_specified_namesequential_input:%!

_user_specified_name19180:%!

_user_specified_name19182:%!

_user_specified_name19184:%!

_user_specified_name19186:%!

_user_specified_name19188:%!

_user_specified_name19190:%!

_user_specified_name19192:%!

_user_specified_name19194:%	!

_user_specified_name19196:%
!

_user_specified_name19198:%!

_user_specified_name19200:%!

_user_specified_name19202:%!

_user_specified_name19204:%!

_user_specified_name19206:%!

_user_specified_name19208:%!

_user_specified_name19210
Т
f
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_19415

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Т
f
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_18961

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
│
№
C__inference_conv2d_6_layer_call_and_return_conditional_losses_19405

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ==@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ==@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         ==@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         ==@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         ?? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         ?? 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
┬C
│
G__inference_sequential_3_layer_call_and_return_conditional_losses_19130
sequential_input 
sequential_1_19000:	 
sequential_1_19002:	(
conv2d_5_19016: 
conv2d_5_19018: (
conv2d_6_19033: @
conv2d_6_19035:@(
conv2d_7_19050:@@
conv2d_7_19052:@(
conv2d_8_19067:@@
conv2d_8_19069:@(
conv2d_9_19084:@@
conv2d_9_19086:@ 
dense_2_19108:	А@
dense_2_19110:@
dense_3_19124:@
dense_3_19126:
identityИв conv2d_5/StatefulPartitionedCallв conv2d_6/StatefulPartitionedCallв conv2d_7/StatefulPartitionedCallв conv2d_8/StatefulPartitionedCallв conv2d_9/StatefulPartitionedCallвdense_2/StatefulPartitionedCallвdense_3/StatefulPartitionedCallв$sequential_1/StatefulPartitionedCall╧
sequential/PartitionedCallPartitionedCallsequential_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_18629а
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall#sequential/PartitionedCall:output:0sequential_1_19000sequential_1_19002*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_18894Ь
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall-sequential_1/StatefulPartitionedCall:output:0conv2d_5_19016conv2d_5_19018*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ~~ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_19015Ё
max_pooling2d_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ?? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_18951Ч
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0conv2d_6_19033conv2d_6_19035*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ==@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_19032Ё
max_pooling2d_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_18961Ч
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0conv2d_7_19050conv2d_7_19052*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_19049Ё
max_pooling2d_7/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_18971Ч
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_7/PartitionedCall:output:0conv2d_8_19067conv2d_8_19069*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_19066Ё
max_pooling2d_8/PartitionedCallPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_18981Ч
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_8/PartitionedCall:output:0conv2d_9_19084conv2d_9_19086*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_19083Ё
max_pooling2d_9/PartitionedCallPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_18991▄
flatten_1/PartitionedCallPartitionedCall(max_pooling2d_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_19095Е
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_19108dense_2_19110*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_19107Л
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_19124dense_3_19126*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_19123w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ╝
NoOpNoOp!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:         АА: : : : : : : : : : : : : : : : 2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall:c _
1
_output_shapes
:         АА
*
_user_specified_namesequential_input:%!

_user_specified_name19000:%!

_user_specified_name19002:%!

_user_specified_name19016:%!

_user_specified_name19018:%!

_user_specified_name19033:%!

_user_specified_name19035:%!

_user_specified_name19050:%!

_user_specified_name19052:%	!

_user_specified_name19067:%
!

_user_specified_name19069:%!

_user_specified_name19084:%!

_user_specified_name19086:%!

_user_specified_name19108:%!

_user_specified_name19110:%!

_user_specified_name19124:%!

_user_specified_name19126
гЧ
■
F__inference_random_flip_layer_call_and_return_conditional_losses_19703

inputs?
1stateful_uniform_full_int_rngreadandskip_resource:	
identityИв(stateful_uniform_full_int/RngReadAndSkipв*stateful_uniform_full_int_1/RngReadAndSkipi
stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:i
stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ы
stateful_uniform_full_int/ProdProd(stateful_uniform_full_int/shape:output:0(stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: b
 stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :Б
 stateful_uniform_full_int/Cast_1Cast'stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ┌
(stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip1stateful_uniform_full_int_rngreadandskip_resource)stateful_uniform_full_int/Cast/x:output:0$stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:w
-stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
'stateful_uniform_full_int/strided_sliceStridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:06stateful_uniform_full_int/strided_slice/stack:output:08stateful_uniform_full_int/strided_slice/stack_1:output:08stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_maskП
!stateful_uniform_full_int/BitcastBitcast0stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0y
/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╧
)stateful_uniform_full_int/strided_slice_1StridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:08stateful_uniform_full_int/strided_slice_1/stack:output:0:stateful_uniform_full_int/strided_slice_1/stack_1:output:0:stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:У
#stateful_uniform_full_int/Bitcast_1Bitcast2stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0_
stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :С
stateful_uniform_full_intStatelessRandomUniformFullIntV2(stateful_uniform_full_int/shape:output:0,stateful_uniform_full_int/Bitcast_1:output:0*stateful_uniform_full_int/Bitcast:output:0&stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	T

zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R x
stackPack"stateful_uniform_full_int:output:0zeros_like:output:0*
N*
T0	*
_output_shapes

:d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ў
strided_sliceStridedSlicestack:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_maskЮ
3stateless_random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*1
_output_shapes
:         ААа
&stateless_random_flip_left_right/ShapeShape<stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
::э╧~
4stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: А
6stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:А
6stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ў
.stateless_random_flip_left_right/strided_sliceStridedSlice/stateless_random_flip_left_right/Shape:output:0=stateless_random_flip_left_right/strided_slice/stack:output:0?stateless_random_flip_left_right/strided_slice/stack_1:output:0?stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskо
?stateless_random_flip_left_right/stateless_random_uniform/shapePack7stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:В
=stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    В
=stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  А?░
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice:output:0* 
_output_shapes
::Ш
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :■
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Hstateless_random_flip_left_right/stateless_random_uniform/shape:output:0\stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0`stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0_stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:         ї
=stateless_random_flip_left_right/stateless_random_uniform/subSubFstateless_random_flip_left_right/stateless_random_uniform/max:output:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: Т
=stateless_random_flip_left_right/stateless_random_uniform/mulMul[stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Astateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:         √
9stateless_random_flip_left_right/stateless_random_uniformAddV2Astateless_random_flip_left_right/stateless_random_uniform/mul:z:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:         r
0stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :r
0stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :r
0stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :╬
.stateless_random_flip_left_right/Reshape/shapePack7stateless_random_flip_left_right/strided_slice:output:09stateless_random_flip_left_right/Reshape/shape/1:output:09stateless_random_flip_left_right/Reshape/shape/2:output:09stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:х
(stateless_random_flip_left_right/ReshapeReshape=stateless_random_flip_left_right/stateless_random_uniform:z:07stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:         Ь
&stateless_random_flip_left_right/RoundRound1stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:         y
/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:ы
*stateless_random_flip_left_right/ReverseV2	ReverseV2<stateless_random_flip_left_right/control_dependency:output:08stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:         АА╚
$stateless_random_flip_left_right/mulMul*stateless_random_flip_left_right/Round:y:03stateless_random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:         ААk
&stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?┬
$stateless_random_flip_left_right/subSub/stateless_random_flip_left_right/sub/x:output:0*stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:         ╤
&stateless_random_flip_left_right/mul_1Mul(stateless_random_flip_left_right/sub:z:0<stateless_random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:         АА┐
$stateless_random_flip_left_right/addAddV2(stateless_random_flip_left_right/mul:z:0*stateless_random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:         ААk
!stateful_uniform_full_int_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:k
!stateful_uniform_full_int_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: б
 stateful_uniform_full_int_1/ProdProd*stateful_uniform_full_int_1/shape:output:0*stateful_uniform_full_int_1/Const:output:0*
T0*
_output_shapes
: d
"stateful_uniform_full_int_1/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :Е
"stateful_uniform_full_int_1/Cast_1Cast)stateful_uniform_full_int_1/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: Л
*stateful_uniform_full_int_1/RngReadAndSkipRngReadAndSkip1stateful_uniform_full_int_rngreadandskip_resource+stateful_uniform_full_int_1/Cast/x:output:0&stateful_uniform_full_int_1/Cast_1:y:0)^stateful_uniform_full_int/RngReadAndSkip*
_output_shapes
:y
/stateful_uniform_full_int_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1stateful_uniform_full_int_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:у
)stateful_uniform_full_int_1/strided_sliceStridedSlice2stateful_uniform_full_int_1/RngReadAndSkip:value:08stateful_uniform_full_int_1/strided_slice/stack:output:0:stateful_uniform_full_int_1/strided_slice/stack_1:output:0:stateful_uniform_full_int_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_maskУ
#stateful_uniform_full_int_1/BitcastBitcast2stateful_uniform_full_int_1/strided_slice:output:0*
T0	*
_output_shapes
:*

type0{
1stateful_uniform_full_int_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3stateful_uniform_full_int_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3stateful_uniform_full_int_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
+stateful_uniform_full_int_1/strided_slice_1StridedSlice2stateful_uniform_full_int_1/RngReadAndSkip:value:0:stateful_uniform_full_int_1/strided_slice_1/stack:output:0<stateful_uniform_full_int_1/strided_slice_1/stack_1:output:0<stateful_uniform_full_int_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:Ч
%stateful_uniform_full_int_1/Bitcast_1Bitcast4stateful_uniform_full_int_1/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0a
stateful_uniform_full_int_1/algConst*
_output_shapes
: *
dtype0*
value	B :Ы
stateful_uniform_full_int_1StatelessRandomUniformFullIntV2*stateful_uniform_full_int_1/shape:output:0.stateful_uniform_full_int_1/Bitcast_1:output:0,stateful_uniform_full_int_1/Bitcast:output:0(stateful_uniform_full_int_1/alg:output:0*
_output_shapes
:*
dtype0	V
zeros_like_1Const*
_output_shapes
:*
dtype0	*
valueB	R ~
stack_1Pack$stateful_uniform_full_int_1:output:0zeros_like_1:output:0*
N*
T0	*
_output_shapes

:f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Б
strided_slice_1StridedSlicestack_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask█
0stateless_random_flip_up_down/control_dependencyIdentity(stateless_random_flip_left_right/add:z:0*
T0*7
_class-
+)loc:@stateless_random_flip_left_right/add*1
_output_shapes
:         ААЪ
#stateless_random_flip_up_down/ShapeShape9stateless_random_flip_up_down/control_dependency:output:0*
T0*
_output_shapes
::э╧{
1stateless_random_flip_up_down/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3stateless_random_flip_up_down/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3stateless_random_flip_up_down/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
+stateless_random_flip_up_down/strided_sliceStridedSlice,stateless_random_flip_up_down/Shape:output:0:stateless_random_flip_up_down/strided_slice/stack:output:0<stateless_random_flip_up_down/strided_slice/stack_1:output:0<stateless_random_flip_up_down/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskи
<stateless_random_flip_up_down/stateless_random_uniform/shapePack4stateless_random_flip_up_down/strided_slice:output:0*
N*
T0*
_output_shapes
:
:stateless_random_flip_up_down/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    
:stateless_random_flip_up_down/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  А?п
Sstateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice_1:output:0* 
_output_shapes
::Х
Sstateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :я
Ostateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Estateless_random_flip_up_down/stateless_random_uniform/shape:output:0Ystateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0]stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0\stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:         ь
:stateless_random_flip_up_down/stateless_random_uniform/subSubCstateless_random_flip_up_down/stateless_random_uniform/max:output:0Cstateless_random_flip_up_down/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: Й
:stateless_random_flip_up_down/stateless_random_uniform/mulMulXstateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2:output:0>stateless_random_flip_up_down/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:         Є
6stateless_random_flip_up_down/stateless_random_uniformAddV2>stateless_random_flip_up_down/stateless_random_uniform/mul:z:0Cstateless_random_flip_up_down/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:         o
-stateless_random_flip_up_down/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :o
-stateless_random_flip_up_down/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :o
-stateless_random_flip_up_down/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :┐
+stateless_random_flip_up_down/Reshape/shapePack4stateless_random_flip_up_down/strided_slice:output:06stateless_random_flip_up_down/Reshape/shape/1:output:06stateless_random_flip_up_down/Reshape/shape/2:output:06stateless_random_flip_up_down/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:▄
%stateless_random_flip_up_down/ReshapeReshape:stateless_random_flip_up_down/stateless_random_uniform:z:04stateless_random_flip_up_down/Reshape/shape:output:0*
T0*/
_output_shapes
:         Ц
#stateless_random_flip_up_down/RoundRound.stateless_random_flip_up_down/Reshape:output:0*
T0*/
_output_shapes
:         v
,stateless_random_flip_up_down/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:т
'stateless_random_flip_up_down/ReverseV2	ReverseV29stateless_random_flip_up_down/control_dependency:output:05stateless_random_flip_up_down/ReverseV2/axis:output:0*
T0*1
_output_shapes
:         АА┐
!stateless_random_flip_up_down/mulMul'stateless_random_flip_up_down/Round:y:00stateless_random_flip_up_down/ReverseV2:output:0*
T0*1
_output_shapes
:         ААh
#stateless_random_flip_up_down/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╣
!stateless_random_flip_up_down/subSub,stateless_random_flip_up_down/sub/x:output:0'stateless_random_flip_up_down/Round:y:0*
T0*/
_output_shapes
:         ╚
#stateless_random_flip_up_down/mul_1Mul%stateless_random_flip_up_down/sub:z:09stateless_random_flip_up_down/control_dependency:output:0*
T0*1
_output_shapes
:         АА╢
!stateless_random_flip_up_down/addAddV2%stateless_random_flip_up_down/mul:z:0'stateless_random_flip_up_down/mul_1:z:0*
T0*1
_output_shapes
:         АА~
IdentityIdentity%stateless_random_flip_up_down/add:z:0^NoOp*
T0*1
_output_shapes
:         ААz
NoOpNoOp)^stateful_uniform_full_int/RngReadAndSkip+^stateful_uniform_full_int_1/RngReadAndSkip*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         АА: 2T
(stateful_uniform_full_int/RngReadAndSkip(stateful_uniform_full_int/RngReadAndSkip2X
*stateful_uniform_full_int_1/RngReadAndSkip*stateful_uniform_full_int_1/RngReadAndSkip:Y U
1
_output_shapes
:         АА
 
_user_specified_nameinputs:($
"
_user_specified_name
resource
│
№
C__inference_conv2d_9_layer_call_and_return_conditional_losses_19083

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
╢
K
/__inference_max_pooling2d_6_layer_call_fn_19410

inputs
identity╪
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_18961Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
ЁЦ
в0
__inference__traced_save_20191
file_prefix@
&read_disablecopyonread_conv2d_5_kernel: 4
&read_1_disablecopyonread_conv2d_5_bias: B
(read_2_disablecopyonread_conv2d_6_kernel: @4
&read_3_disablecopyonread_conv2d_6_bias:@B
(read_4_disablecopyonread_conv2d_7_kernel:@@4
&read_5_disablecopyonread_conv2d_7_bias:@B
(read_6_disablecopyonread_conv2d_8_kernel:@@4
&read_7_disablecopyonread_conv2d_8_bias:@B
(read_8_disablecopyonread_conv2d_9_kernel:@@4
&read_9_disablecopyonread_conv2d_9_bias:@;
(read_10_disablecopyonread_dense_2_kernel:	А@4
&read_11_disablecopyonread_dense_2_bias:@:
(read_12_disablecopyonread_dense_3_kernel:@4
&read_13_disablecopyonread_dense_3_bias:-
#read_14_disablecopyonread_adam_iter:	 /
%read_15_disablecopyonread_adam_beta_1: /
%read_16_disablecopyonread_adam_beta_2: .
$read_17_disablecopyonread_adam_decay: 6
,read_18_disablecopyonread_adam_learning_rate: +
!read_19_disablecopyonread_total_1: +
!read_20_disablecopyonread_count_1: )
read_21_disablecopyonread_total: )
read_22_disablecopyonread_count: <
.read_23_disablecopyonread_random_flip_statevar:	@
2read_24_disablecopyonread_random_rotation_statevar:	J
0read_25_disablecopyonread_adam_conv2d_5_kernel_m: <
.read_26_disablecopyonread_adam_conv2d_5_bias_m: J
0read_27_disablecopyonread_adam_conv2d_6_kernel_m: @<
.read_28_disablecopyonread_adam_conv2d_6_bias_m:@J
0read_29_disablecopyonread_adam_conv2d_7_kernel_m:@@<
.read_30_disablecopyonread_adam_conv2d_7_bias_m:@J
0read_31_disablecopyonread_adam_conv2d_8_kernel_m:@@<
.read_32_disablecopyonread_adam_conv2d_8_bias_m:@J
0read_33_disablecopyonread_adam_conv2d_9_kernel_m:@@<
.read_34_disablecopyonread_adam_conv2d_9_bias_m:@B
/read_35_disablecopyonread_adam_dense_2_kernel_m:	А@;
-read_36_disablecopyonread_adam_dense_2_bias_m:@A
/read_37_disablecopyonread_adam_dense_3_kernel_m:@;
-read_38_disablecopyonread_adam_dense_3_bias_m:J
0read_39_disablecopyonread_adam_conv2d_5_kernel_v: <
.read_40_disablecopyonread_adam_conv2d_5_bias_v: J
0read_41_disablecopyonread_adam_conv2d_6_kernel_v: @<
.read_42_disablecopyonread_adam_conv2d_6_bias_v:@J
0read_43_disablecopyonread_adam_conv2d_7_kernel_v:@@<
.read_44_disablecopyonread_adam_conv2d_7_bias_v:@J
0read_45_disablecopyonread_adam_conv2d_8_kernel_v:@@<
.read_46_disablecopyonread_adam_conv2d_8_bias_v:@J
0read_47_disablecopyonread_adam_conv2d_9_kernel_v:@@<
.read_48_disablecopyonread_adam_conv2d_9_bias_v:@B
/read_49_disablecopyonread_adam_dense_2_kernel_v:	А@;
-read_50_disablecopyonread_adam_dense_2_bias_v:@A
/read_51_disablecopyonread_adam_dense_3_kernel_v:@;
-read_52_disablecopyonread_adam_dense_3_bias_v:
savev2_const
identity_107ИвMergeV2CheckpointsвRead/DisableCopyOnReadвRead/ReadVariableOpвRead_1/DisableCopyOnReadвRead_1/ReadVariableOpвRead_10/DisableCopyOnReadвRead_10/ReadVariableOpвRead_11/DisableCopyOnReadвRead_11/ReadVariableOpвRead_12/DisableCopyOnReadвRead_12/ReadVariableOpвRead_13/DisableCopyOnReadвRead_13/ReadVariableOpвRead_14/DisableCopyOnReadвRead_14/ReadVariableOpвRead_15/DisableCopyOnReadвRead_15/ReadVariableOpвRead_16/DisableCopyOnReadвRead_16/ReadVariableOpвRead_17/DisableCopyOnReadвRead_17/ReadVariableOpвRead_18/DisableCopyOnReadвRead_18/ReadVariableOpвRead_19/DisableCopyOnReadвRead_19/ReadVariableOpвRead_2/DisableCopyOnReadвRead_2/ReadVariableOpвRead_20/DisableCopyOnReadвRead_20/ReadVariableOpвRead_21/DisableCopyOnReadвRead_21/ReadVariableOpвRead_22/DisableCopyOnReadвRead_22/ReadVariableOpвRead_23/DisableCopyOnReadвRead_23/ReadVariableOpвRead_24/DisableCopyOnReadвRead_24/ReadVariableOpвRead_25/DisableCopyOnReadвRead_25/ReadVariableOpвRead_26/DisableCopyOnReadвRead_26/ReadVariableOpвRead_27/DisableCopyOnReadвRead_27/ReadVariableOpвRead_28/DisableCopyOnReadвRead_28/ReadVariableOpвRead_29/DisableCopyOnReadвRead_29/ReadVariableOpвRead_3/DisableCopyOnReadвRead_3/ReadVariableOpвRead_30/DisableCopyOnReadвRead_30/ReadVariableOpвRead_31/DisableCopyOnReadвRead_31/ReadVariableOpвRead_32/DisableCopyOnReadвRead_32/ReadVariableOpвRead_33/DisableCopyOnReadвRead_33/ReadVariableOpвRead_34/DisableCopyOnReadвRead_34/ReadVariableOpвRead_35/DisableCopyOnReadвRead_35/ReadVariableOpвRead_36/DisableCopyOnReadвRead_36/ReadVariableOpвRead_37/DisableCopyOnReadвRead_37/ReadVariableOpвRead_38/DisableCopyOnReadвRead_38/ReadVariableOpвRead_39/DisableCopyOnReadвRead_39/ReadVariableOpвRead_4/DisableCopyOnReadвRead_4/ReadVariableOpвRead_40/DisableCopyOnReadвRead_40/ReadVariableOpвRead_41/DisableCopyOnReadвRead_41/ReadVariableOpвRead_42/DisableCopyOnReadвRead_42/ReadVariableOpвRead_43/DisableCopyOnReadвRead_43/ReadVariableOpвRead_44/DisableCopyOnReadвRead_44/ReadVariableOpвRead_45/DisableCopyOnReadвRead_45/ReadVariableOpвRead_46/DisableCopyOnReadвRead_46/ReadVariableOpвRead_47/DisableCopyOnReadвRead_47/ReadVariableOpвRead_48/DisableCopyOnReadвRead_48/ReadVariableOpвRead_49/DisableCopyOnReadвRead_49/ReadVariableOpвRead_5/DisableCopyOnReadвRead_5/ReadVariableOpвRead_50/DisableCopyOnReadвRead_50/ReadVariableOpвRead_51/DisableCopyOnReadвRead_51/ReadVariableOpвRead_52/DisableCopyOnReadвRead_52/ReadVariableOpвRead_6/DisableCopyOnReadвRead_6/ReadVariableOpвRead_7/DisableCopyOnReadвRead_7/ReadVariableOpвRead_8/DisableCopyOnReadвRead_8/ReadVariableOpвRead_9/DisableCopyOnReadвRead_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: x
Read/DisableCopyOnReadDisableCopyOnRead&read_disablecopyonread_conv2d_5_kernel"/device:CPU:0*
_output_shapes
 к
Read/ReadVariableOpReadVariableOp&read_disablecopyonread_conv2d_5_kernel^Read/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0q
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
: z
Read_1/DisableCopyOnReadDisableCopyOnRead&read_1_disablecopyonread_conv2d_5_bias"/device:CPU:0*
_output_shapes
 в
Read_1/ReadVariableOpReadVariableOp&read_1_disablecopyonread_conv2d_5_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
: |
Read_2/DisableCopyOnReadDisableCopyOnRead(read_2_disablecopyonread_conv2d_6_kernel"/device:CPU:0*
_output_shapes
 ░
Read_2/ReadVariableOpReadVariableOp(read_2_disablecopyonread_conv2d_6_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0u

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @k

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*&
_output_shapes
: @z
Read_3/DisableCopyOnReadDisableCopyOnRead&read_3_disablecopyonread_conv2d_6_bias"/device:CPU:0*
_output_shapes
 в
Read_3/ReadVariableOpReadVariableOp&read_3_disablecopyonread_conv2d_6_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:@|
Read_4/DisableCopyOnReadDisableCopyOnRead(read_4_disablecopyonread_conv2d_7_kernel"/device:CPU:0*
_output_shapes
 ░
Read_4/ReadVariableOpReadVariableOp(read_4_disablecopyonread_conv2d_7_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0u

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@k

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@z
Read_5/DisableCopyOnReadDisableCopyOnRead&read_5_disablecopyonread_conv2d_7_bias"/device:CPU:0*
_output_shapes
 в
Read_5/ReadVariableOpReadVariableOp&read_5_disablecopyonread_conv2d_7_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:@|
Read_6/DisableCopyOnReadDisableCopyOnRead(read_6_disablecopyonread_conv2d_8_kernel"/device:CPU:0*
_output_shapes
 ░
Read_6/ReadVariableOpReadVariableOp(read_6_disablecopyonread_conv2d_8_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0v
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@z
Read_7/DisableCopyOnReadDisableCopyOnRead&read_7_disablecopyonread_conv2d_8_bias"/device:CPU:0*
_output_shapes
 в
Read_7/ReadVariableOpReadVariableOp&read_7_disablecopyonread_conv2d_8_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:@|
Read_8/DisableCopyOnReadDisableCopyOnRead(read_8_disablecopyonread_conv2d_9_kernel"/device:CPU:0*
_output_shapes
 ░
Read_8/ReadVariableOpReadVariableOp(read_8_disablecopyonread_conv2d_9_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0v
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@z
Read_9/DisableCopyOnReadDisableCopyOnRead&read_9_disablecopyonread_conv2d_9_bias"/device:CPU:0*
_output_shapes
 в
Read_9/ReadVariableOpReadVariableOp&read_9_disablecopyonread_conv2d_9_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:@}
Read_10/DisableCopyOnReadDisableCopyOnRead(read_10_disablecopyonread_dense_2_kernel"/device:CPU:0*
_output_shapes
 л
Read_10/ReadVariableOpReadVariableOp(read_10_disablecopyonread_dense_2_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	А@*
dtype0p
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	А@f
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:	А@{
Read_11/DisableCopyOnReadDisableCopyOnRead&read_11_disablecopyonread_dense_2_bias"/device:CPU:0*
_output_shapes
 д
Read_11/ReadVariableOpReadVariableOp&read_11_disablecopyonread_dense_2_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:@}
Read_12/DisableCopyOnReadDisableCopyOnRead(read_12_disablecopyonread_dense_3_kernel"/device:CPU:0*
_output_shapes
 к
Read_12/ReadVariableOpReadVariableOp(read_12_disablecopyonread_dense_3_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes

:@{
Read_13/DisableCopyOnReadDisableCopyOnRead&read_13_disablecopyonread_dense_3_bias"/device:CPU:0*
_output_shapes
 д
Read_13/ReadVariableOpReadVariableOp&read_13_disablecopyonread_dense_3_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_14/DisableCopyOnReadDisableCopyOnRead#read_14_disablecopyonread_adam_iter"/device:CPU:0*
_output_shapes
 Э
Read_14/ReadVariableOpReadVariableOp#read_14_disablecopyonread_adam_iter^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0	*
_output_shapes
: z
Read_15/DisableCopyOnReadDisableCopyOnRead%read_15_disablecopyonread_adam_beta_1"/device:CPU:0*
_output_shapes
 Я
Read_15/ReadVariableOpReadVariableOp%read_15_disablecopyonread_adam_beta_1^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
: z
Read_16/DisableCopyOnReadDisableCopyOnRead%read_16_disablecopyonread_adam_beta_2"/device:CPU:0*
_output_shapes
 Я
Read_16/ReadVariableOpReadVariableOp%read_16_disablecopyonread_adam_beta_2^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
: y
Read_17/DisableCopyOnReadDisableCopyOnRead$read_17_disablecopyonread_adam_decay"/device:CPU:0*
_output_shapes
 Ю
Read_17/ReadVariableOpReadVariableOp$read_17_disablecopyonread_adam_decay^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
: Б
Read_18/DisableCopyOnReadDisableCopyOnRead,read_18_disablecopyonread_adam_learning_rate"/device:CPU:0*
_output_shapes
 ж
Read_18/ReadVariableOpReadVariableOp,read_18_disablecopyonread_adam_learning_rate^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_19/DisableCopyOnReadDisableCopyOnRead!read_19_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 Ы
Read_19/ReadVariableOpReadVariableOp!read_19_disablecopyonread_total_1^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_20/DisableCopyOnReadDisableCopyOnRead!read_20_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 Ы
Read_20/ReadVariableOpReadVariableOp!read_20_disablecopyonread_count_1^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_21/DisableCopyOnReadDisableCopyOnReadread_21_disablecopyonread_total"/device:CPU:0*
_output_shapes
 Щ
Read_21/ReadVariableOpReadVariableOpread_21_disablecopyonread_total^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_22/DisableCopyOnReadDisableCopyOnReadread_22_disablecopyonread_count"/device:CPU:0*
_output_shapes
 Щ
Read_22/ReadVariableOpReadVariableOpread_22_disablecopyonread_count^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
: Г
Read_23/DisableCopyOnReadDisableCopyOnRead.read_23_disablecopyonread_random_flip_statevar"/device:CPU:0*
_output_shapes
 м
Read_23/ReadVariableOpReadVariableOp.read_23_disablecopyonread_random_flip_statevar^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0	k
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
:a
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0	*
_output_shapes
:З
Read_24/DisableCopyOnReadDisableCopyOnRead2read_24_disablecopyonread_random_rotation_statevar"/device:CPU:0*
_output_shapes
 ░
Read_24/ReadVariableOpReadVariableOp2read_24_disablecopyonread_random_rotation_statevar^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0	k
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
:a
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0	*
_output_shapes
:Е
Read_25/DisableCopyOnReadDisableCopyOnRead0read_25_disablecopyonread_adam_conv2d_5_kernel_m"/device:CPU:0*
_output_shapes
 ║
Read_25/ReadVariableOpReadVariableOp0read_25_disablecopyonread_adam_conv2d_5_kernel_m^Read_25/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*&
_output_shapes
: Г
Read_26/DisableCopyOnReadDisableCopyOnRead.read_26_disablecopyonread_adam_conv2d_5_bias_m"/device:CPU:0*
_output_shapes
 м
Read_26/ReadVariableOpReadVariableOp.read_26_disablecopyonread_adam_conv2d_5_bias_m^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
: Е
Read_27/DisableCopyOnReadDisableCopyOnRead0read_27_disablecopyonread_adam_conv2d_6_kernel_m"/device:CPU:0*
_output_shapes
 ║
Read_27/ReadVariableOpReadVariableOp0read_27_disablecopyonread_adam_conv2d_6_kernel_m^Read_27/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0w
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @m
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*&
_output_shapes
: @Г
Read_28/DisableCopyOnReadDisableCopyOnRead.read_28_disablecopyonread_adam_conv2d_6_bias_m"/device:CPU:0*
_output_shapes
 м
Read_28/ReadVariableOpReadVariableOp.read_28_disablecopyonread_adam_conv2d_6_bias_m^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
:@Е
Read_29/DisableCopyOnReadDisableCopyOnRead0read_29_disablecopyonread_adam_conv2d_7_kernel_m"/device:CPU:0*
_output_shapes
 ║
Read_29/ReadVariableOpReadVariableOp0read_29_disablecopyonread_adam_conv2d_7_kernel_m^Read_29/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0w
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@Г
Read_30/DisableCopyOnReadDisableCopyOnRead.read_30_disablecopyonread_adam_conv2d_7_bias_m"/device:CPU:0*
_output_shapes
 м
Read_30/ReadVariableOpReadVariableOp.read_30_disablecopyonread_adam_conv2d_7_bias_m^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
:@Е
Read_31/DisableCopyOnReadDisableCopyOnRead0read_31_disablecopyonread_adam_conv2d_8_kernel_m"/device:CPU:0*
_output_shapes
 ║
Read_31/ReadVariableOpReadVariableOp0read_31_disablecopyonread_adam_conv2d_8_kernel_m^Read_31/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0w
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@Г
Read_32/DisableCopyOnReadDisableCopyOnRead.read_32_disablecopyonread_adam_conv2d_8_bias_m"/device:CPU:0*
_output_shapes
 м
Read_32/ReadVariableOpReadVariableOp.read_32_disablecopyonread_adam_conv2d_8_bias_m^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes
:@Е
Read_33/DisableCopyOnReadDisableCopyOnRead0read_33_disablecopyonread_adam_conv2d_9_kernel_m"/device:CPU:0*
_output_shapes
 ║
Read_33/ReadVariableOpReadVariableOp0read_33_disablecopyonread_adam_conv2d_9_kernel_m^Read_33/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0w
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@Г
Read_34/DisableCopyOnReadDisableCopyOnRead.read_34_disablecopyonread_adam_conv2d_9_bias_m"/device:CPU:0*
_output_shapes
 м
Read_34/ReadVariableOpReadVariableOp.read_34_disablecopyonread_adam_conv2d_9_bias_m^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes
:@Д
Read_35/DisableCopyOnReadDisableCopyOnRead/read_35_disablecopyonread_adam_dense_2_kernel_m"/device:CPU:0*
_output_shapes
 ▓
Read_35/ReadVariableOpReadVariableOp/read_35_disablecopyonread_adam_dense_2_kernel_m^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	А@*
dtype0p
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	А@f
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
:	А@В
Read_36/DisableCopyOnReadDisableCopyOnRead-read_36_disablecopyonread_adam_dense_2_bias_m"/device:CPU:0*
_output_shapes
 л
Read_36/ReadVariableOpReadVariableOp-read_36_disablecopyonread_adam_dense_2_bias_m^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes
:@Д
Read_37/DisableCopyOnReadDisableCopyOnRead/read_37_disablecopyonread_adam_dense_3_kernel_m"/device:CPU:0*
_output_shapes
 ▒
Read_37/ReadVariableOpReadVariableOp/read_37_disablecopyonread_adam_dense_3_kernel_m^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes

:@В
Read_38/DisableCopyOnReadDisableCopyOnRead-read_38_disablecopyonread_adam_dense_3_bias_m"/device:CPU:0*
_output_shapes
 л
Read_38/ReadVariableOpReadVariableOp-read_38_disablecopyonread_adam_dense_3_bias_m^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes
:Е
Read_39/DisableCopyOnReadDisableCopyOnRead0read_39_disablecopyonread_adam_conv2d_5_kernel_v"/device:CPU:0*
_output_shapes
 ║
Read_39/ReadVariableOpReadVariableOp0read_39_disablecopyonread_adam_conv2d_5_kernel_v^Read_39/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*&
_output_shapes
: Г
Read_40/DisableCopyOnReadDisableCopyOnRead.read_40_disablecopyonread_adam_conv2d_5_bias_v"/device:CPU:0*
_output_shapes
 м
Read_40/ReadVariableOpReadVariableOp.read_40_disablecopyonread_adam_conv2d_5_bias_v^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes
: Е
Read_41/DisableCopyOnReadDisableCopyOnRead0read_41_disablecopyonread_adam_conv2d_6_kernel_v"/device:CPU:0*
_output_shapes
 ║
Read_41/ReadVariableOpReadVariableOp0read_41_disablecopyonread_adam_conv2d_6_kernel_v^Read_41/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0w
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @m
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*&
_output_shapes
: @Г
Read_42/DisableCopyOnReadDisableCopyOnRead.read_42_disablecopyonread_adam_conv2d_6_bias_v"/device:CPU:0*
_output_shapes
 м
Read_42/ReadVariableOpReadVariableOp.read_42_disablecopyonread_adam_conv2d_6_bias_v^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes
:@Е
Read_43/DisableCopyOnReadDisableCopyOnRead0read_43_disablecopyonread_adam_conv2d_7_kernel_v"/device:CPU:0*
_output_shapes
 ║
Read_43/ReadVariableOpReadVariableOp0read_43_disablecopyonread_adam_conv2d_7_kernel_v^Read_43/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0w
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@Г
Read_44/DisableCopyOnReadDisableCopyOnRead.read_44_disablecopyonread_adam_conv2d_7_bias_v"/device:CPU:0*
_output_shapes
 м
Read_44/ReadVariableOpReadVariableOp.read_44_disablecopyonread_adam_conv2d_7_bias_v^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes
:@Е
Read_45/DisableCopyOnReadDisableCopyOnRead0read_45_disablecopyonread_adam_conv2d_8_kernel_v"/device:CPU:0*
_output_shapes
 ║
Read_45/ReadVariableOpReadVariableOp0read_45_disablecopyonread_adam_conv2d_8_kernel_v^Read_45/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0w
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@Г
Read_46/DisableCopyOnReadDisableCopyOnRead.read_46_disablecopyonread_adam_conv2d_8_bias_v"/device:CPU:0*
_output_shapes
 м
Read_46/ReadVariableOpReadVariableOp.read_46_disablecopyonread_adam_conv2d_8_bias_v^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes
:@Е
Read_47/DisableCopyOnReadDisableCopyOnRead0read_47_disablecopyonread_adam_conv2d_9_kernel_v"/device:CPU:0*
_output_shapes
 ║
Read_47/ReadVariableOpReadVariableOp0read_47_disablecopyonread_adam_conv2d_9_kernel_v^Read_47/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0w
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@Г
Read_48/DisableCopyOnReadDisableCopyOnRead.read_48_disablecopyonread_adam_conv2d_9_bias_v"/device:CPU:0*
_output_shapes
 м
Read_48/ReadVariableOpReadVariableOp.read_48_disablecopyonread_adam_conv2d_9_bias_v^Read_48/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes
:@Д
Read_49/DisableCopyOnReadDisableCopyOnRead/read_49_disablecopyonread_adam_dense_2_kernel_v"/device:CPU:0*
_output_shapes
 ▓
Read_49/ReadVariableOpReadVariableOp/read_49_disablecopyonread_adam_dense_2_kernel_v^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	А@*
dtype0p
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	А@f
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes
:	А@В
Read_50/DisableCopyOnReadDisableCopyOnRead-read_50_disablecopyonread_adam_dense_2_bias_v"/device:CPU:0*
_output_shapes
 л
Read_50/ReadVariableOpReadVariableOp-read_50_disablecopyonread_adam_dense_2_bias_v^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes
:@Д
Read_51/DisableCopyOnReadDisableCopyOnRead/read_51_disablecopyonread_adam_dense_3_kernel_v"/device:CPU:0*
_output_shapes
 ▒
Read_51/ReadVariableOpReadVariableOp/read_51_disablecopyonread_adam_dense_3_kernel_v^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0p
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@g
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes

:@В
Read_52/DisableCopyOnReadDisableCopyOnRead-read_52_disablecopyonread_adam_dense_3_bias_v"/device:CPU:0*
_output_shapes
 л
Read_52/ReadVariableOpReadVariableOp-read_52_disablecopyonread_adam_dense_3_bias_v^Read_52/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes
:Е
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*о
valueдBб6B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer-1/layer-0/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBRlayer-1/layer-1/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH┘
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*
valuevBt6B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Щ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *D
dtypes:
826			Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:│
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_106Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_107IdentityIdentity_106:output:0^NoOp*
T0*
_output_shapes
: Ш
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "%
identity_107Identity_107:output:0*(
_construction_contextkEagerRuntime*Б
_input_shapesp
n: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:/+
)
_user_specified_nameconv2d_5/kernel:-)
'
_user_specified_nameconv2d_5/bias:/+
)
_user_specified_nameconv2d_6/kernel:-)
'
_user_specified_nameconv2d_6/bias:/+
)
_user_specified_nameconv2d_7/kernel:-)
'
_user_specified_nameconv2d_7/bias:/+
)
_user_specified_nameconv2d_8/kernel:-)
'
_user_specified_nameconv2d_8/bias:/	+
)
_user_specified_nameconv2d_9/kernel:-
)
'
_user_specified_nameconv2d_9/bias:.*
(
_user_specified_namedense_2/kernel:,(
&
_user_specified_namedense_2/bias:.*
(
_user_specified_namedense_3/kernel:,(
&
_user_specified_namedense_3/bias:)%
#
_user_specified_name	Adam/iter:+'
%
_user_specified_nameAdam/beta_1:+'
%
_user_specified_nameAdam/beta_2:*&
$
_user_specified_name
Adam/decay:2.
,
_user_specified_nameAdam/learning_rate:'#
!
_user_specified_name	total_1:'#
!
_user_specified_name	count_1:%!

_user_specified_nametotal:%!

_user_specified_namecount:40
.
_user_specified_namerandom_flip/StateVar:84
2
_user_specified_namerandom_rotation/StateVar:62
0
_user_specified_nameAdam/conv2d_5/kernel/m:40
.
_user_specified_nameAdam/conv2d_5/bias/m:62
0
_user_specified_nameAdam/conv2d_6/kernel/m:40
.
_user_specified_nameAdam/conv2d_6/bias/m:62
0
_user_specified_nameAdam/conv2d_7/kernel/m:40
.
_user_specified_nameAdam/conv2d_7/bias/m:6 2
0
_user_specified_nameAdam/conv2d_8/kernel/m:4!0
.
_user_specified_nameAdam/conv2d_8/bias/m:6"2
0
_user_specified_nameAdam/conv2d_9/kernel/m:4#0
.
_user_specified_nameAdam/conv2d_9/bias/m:5$1
/
_user_specified_nameAdam/dense_2/kernel/m:3%/
-
_user_specified_nameAdam/dense_2/bias/m:5&1
/
_user_specified_nameAdam/dense_3/kernel/m:3'/
-
_user_specified_nameAdam/dense_3/bias/m:6(2
0
_user_specified_nameAdam/conv2d_5/kernel/v:4)0
.
_user_specified_nameAdam/conv2d_5/bias/v:6*2
0
_user_specified_nameAdam/conv2d_6/kernel/v:4+0
.
_user_specified_nameAdam/conv2d_6/bias/v:6,2
0
_user_specified_nameAdam/conv2d_7/kernel/v:4-0
.
_user_specified_nameAdam/conv2d_7/bias/v:6.2
0
_user_specified_nameAdam/conv2d_8/kernel/v:4/0
.
_user_specified_nameAdam/conv2d_8/bias/v:602
0
_user_specified_nameAdam/conv2d_9/kernel/v:410
.
_user_specified_nameAdam/conv2d_9/bias/v:521
/
_user_specified_nameAdam/dense_2/kernel/v:33/
-
_user_specified_nameAdam/dense_2/bias/v:541
/
_user_specified_nameAdam/dense_3/kernel/v:35/
-
_user_specified_nameAdam/dense_3/bias/v:=69

_output_shapes
: 

_user_specified_nameConst
Т
f
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_19385

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
├
D
(__inference_resizing_layer_call_fn_19561

inputs
identity╕
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_resizing_layer_call_and_return_conditional_losses_18617j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         АА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         АА:Y U
1
_output_shapes
:         АА
 
_user_specified_nameinputs
╢
K
/__inference_max_pooling2d_8_layer_call_fn_19470

inputs
identity╪
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_18981Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
я
К
#__inference_signature_wrapper_19355
sequential_input!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@#
	unknown_7:@@
	unknown_8:@
	unknown_9:	А@

unknown_10:@

unknown_11:@

unknown_12:
identityИвStatefulPartitionedCall▐
StatefulPartitionedCallStatefulPartitionedCallsequential_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *)
f$R"
 __inference__wrapped_model_18609o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:         АА: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
1
_output_shapes
:         АА
*
_user_specified_namesequential_input:%!

_user_specified_name19325:%!

_user_specified_name19327:%!

_user_specified_name19329:%!

_user_specified_name19331:%!

_user_specified_name19333:%!

_user_specified_name19335:%!

_user_specified_name19337:%!

_user_specified_name19339:%	!

_user_specified_name19341:%
!

_user_specified_name19343:%!

_user_specified_name19345:%!

_user_specified_name19347:%!

_user_specified_name19349:%!

_user_specified_name19351
Ц
Э
(__inference_conv2d_5_layer_call_fn_19364

inputs!
unknown: 
	unknown_0: 
identityИвStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ~~ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_19015w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         ~~ <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         АА: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         АА
 
_user_specified_nameinputs:%!

_user_specified_name19358:%!

_user_specified_name19360
Т
f
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_18951

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
┼А
├
J__inference_random_rotation_layer_call_and_return_conditional_losses_18889

inputs6
(stateful_uniform_rngreadandskip_resource:	
identityИвstateful_uniform/RngReadAndSkipI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
¤        j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
■        a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
■        j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: d
stateful_uniform/shapePackstrided_slice:output:0*
N*
T0*
_output_shapes
:Y
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|┘а┐Y
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|┘а?`
stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: А
stateful_uniform/ProdProdstateful_uniform/shape:output:0stateful_uniform/Const:output:0*
T0*
_output_shapes
: Y
stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :o
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ╢
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:n
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:м
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask}
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0p
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:в
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:Б
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0o
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :Л
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:         z
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: Ч
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*#
_output_shapes
:         А
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*#
_output_shapes
:         Z
rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?g
rotation_matrix/subSub
Cast_1:y:0rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: ^
rotation_matrix/CosCosstateful_uniform:z:0*
T0*#
_output_shapes
:         \
rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?k
rotation_matrix/sub_1Sub
Cast_1:y:0 rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: |
rotation_matrix/mulMulrotation_matrix/Cos:y:0rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:         ^
rotation_matrix/SinSinstateful_uniform:z:0*
T0*#
_output_shapes
:         \
rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?i
rotation_matrix/sub_2SubCast:y:0 rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: ~
rotation_matrix/mul_1Mulrotation_matrix/Sin:y:0rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:         ~
rotation_matrix/sub_3Subrotation_matrix/mul:z:0rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:         ~
rotation_matrix/sub_4Subrotation_matrix/sub:z:0rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:         ^
rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @П
rotation_matrix/truedivRealDivrotation_matrix/sub_4:z:0"rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:         \
rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?i
rotation_matrix/sub_5SubCast:y:0 rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: `
rotation_matrix/Sin_1Sinstateful_uniform:z:0*
T0*#
_output_shapes
:         \
rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?k
rotation_matrix/sub_6Sub
Cast_1:y:0 rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: А
rotation_matrix/mul_2Mulrotation_matrix/Sin_1:y:0rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:         `
rotation_matrix/Cos_1Cosstateful_uniform:z:0*
T0*#
_output_shapes
:         \
rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?i
rotation_matrix/sub_7SubCast:y:0 rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: А
rotation_matrix/mul_3Mulrotation_matrix/Cos_1:y:0rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:         А
rotation_matrix/addAddV2rotation_matrix/mul_2:z:0rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:         ~
rotation_matrix/sub_8Subrotation_matrix/sub_5:z:0rotation_matrix/add:z:0*
T0*#
_output_shapes
:         `
rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @У
rotation_matrix/truediv_1RealDivrotation_matrix/sub_8:z:0$rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:         g
rotation_matrix/ShapeShapestateful_uniform:z:0*
T0*
_output_shapes
::э╧m
#rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
rotation_matrix/strided_sliceStridedSlicerotation_matrix/Shape:output:0,rotation_matrix/strided_slice/stack:output:0.rotation_matrix/strided_slice/stack_1:output:0.rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
rotation_matrix/Cos_2Cosstateful_uniform:z:0*
T0*#
_output_shapes
:         v
%rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
rotation_matrix/strided_slice_1StridedSlicerotation_matrix/Cos_2:y:0.rotation_matrix/strided_slice_1/stack:output:00rotation_matrix/strided_slice_1/stack_1:output:00rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/Sin_2Sinstateful_uniform:z:0*
T0*#
_output_shapes
:         v
%rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
rotation_matrix/strided_slice_2StridedSlicerotation_matrix/Sin_2:y:0.rotation_matrix/strided_slice_2/stack:output:00rotation_matrix/strided_slice_2/stack_1:output:00rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_maskv
rotation_matrix/NegNeg(rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:         v
%rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╓
rotation_matrix/strided_slice_3StridedSlicerotation_matrix/truediv:z:0.rotation_matrix/strided_slice_3/stack:output:00rotation_matrix/strided_slice_3/stack_1:output:00rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/Sin_3Sinstateful_uniform:z:0*
T0*#
_output_shapes
:         v
%rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
rotation_matrix/strided_slice_4StridedSlicerotation_matrix/Sin_3:y:0.rotation_matrix/strided_slice_4/stack:output:00rotation_matrix/strided_slice_4/stack_1:output:00rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/Cos_3Cosstateful_uniform:z:0*
T0*#
_output_shapes
:         v
%rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╘
rotation_matrix/strided_slice_5StridedSlicerotation_matrix/Cos_3:y:0.rotation_matrix/strided_slice_5/stack:output:00rotation_matrix/strided_slice_5/stack_1:output:00rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_maskv
%rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╪
rotation_matrix/strided_slice_6StridedSlicerotation_matrix/truediv_1:z:0.rotation_matrix/strided_slice_6/stack:output:00rotation_matrix/strided_slice_6/stack_1:output:00rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :г
rotation_matrix/zeros/packedPack&rotation_matrix/strided_slice:output:0'rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:`
rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ь
rotation_matrix/zerosFill%rotation_matrix/zeros/packed:output:0$rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:         ]
rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :О
rotation_matrix/concatConcatV2(rotation_matrix/strided_slice_1:output:0rotation_matrix/Neg:y:0(rotation_matrix/strided_slice_3:output:0(rotation_matrix/strided_slice_4:output:0(rotation_matrix/strided_slice_5:output:0(rotation_matrix/strided_slice_6:output:0rotation_matrix/zeros:output:0$rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:         S
transform/ShapeShapeinputs*
T0*
_output_shapes
::э╧g
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:я
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:Y
transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    б
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputsrotation_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*1
_output_shapes
:         АА*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEARТ
IdentityIdentity9transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:         ААD
NoOpNoOp ^stateful_uniform/RngReadAndSkip*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         АА: 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:         АА
 
_user_specified_nameinputs:($
"
_user_specified_name
resource
╬

є
B__inference_dense_3_layer_call_and_return_conditional_losses_19556

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ь
i
E__inference_sequential_layer_call_and_return_conditional_losses_18629
resizing_input
identity╔
resizing/PartitionedCallPartitionedCallresizing_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_resizing_layer_call_and_return_conditional_losses_18617▐
rescaling/PartitionedCallPartitionedCall!resizing/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_rescaling_layer_call_and_return_conditional_losses_18626t
IdentityIdentity"rescaling/PartitionedCall:output:0*
T0*1
_output_shapes
:         АА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         АА:a ]
1
_output_shapes
:         АА
(
_user_specified_nameresizing_input
╤
K
/__inference_random_rotation_layer_call_fn_19719

inputs
identity┐
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_random_rotation_layer_call_and_return_conditional_losses_18905j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         АА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         АА:Y U
1
_output_shapes
:         АА
 
_user_specified_nameinputs
╞
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_19516

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         АY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Т
f
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_19505

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ф
b
F__inference_random_flip_layer_call_and_return_conditional_losses_18900

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:         АА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         АА:Y U
1
_output_shapes
:         АА
 
_user_specified_nameinputs
ь
i
E__inference_sequential_layer_call_and_return_conditional_losses_18635
resizing_input
identity╔
resizing/PartitionedCallPartitionedCallresizing_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_resizing_layer_call_and_return_conditional_losses_18617▐
rescaling/PartitionedCallPartitionedCall!resizing/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_rescaling_layer_call_and_return_conditional_losses_18626t
IdentityIdentity"rescaling/PartitionedCall:output:0*
T0*1
_output_shapes
:         АА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         АА:a ]
1
_output_shapes
:         АА
(
_user_specified_nameresizing_input
ь
S
,__inference_sequential_1_layer_call_fn_18922
random_flip_input
identity╟
PartitionedCallPartitionedCallrandom_flip_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_18908j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         АА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         АА:d `
1
_output_shapes
:         АА
+
_user_specified_namerandom_flip_input
Ш
f
J__inference_random_rotation_layer_call_and_return_conditional_losses_19841

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:         АА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         АА:Y U
1
_output_shapes
:         АА
 
_user_specified_nameinputs
Т
Э
(__inference_conv2d_7_layer_call_fn_19424

inputs!
unknown:@@
	unknown_0:@
identityИвStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_19049w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs:%!

_user_specified_name19418:%!

_user_specified_name19420
╞
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_19095

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         АY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
┴
{
+__inference_random_flip_layer_call_fn_19587

inputs
unknown:	
identityИвStatefulPartitionedCall╒
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_random_flip_layer_call_and_return_conditional_losses_18768y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         АА<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         АА: 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         АА
 
_user_specified_nameinputs:%!

_user_specified_name19583
│
а
,__inference_sequential_1_layer_call_fn_18917
random_flip_input
unknown:	
	unknown_0:	
identityИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallrandom_flip_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_18894y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         АА<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         АА: : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
1
_output_shapes
:         АА
+
_user_specified_namerandom_flip_input:%!

_user_specified_name18911:%!

_user_specified_name18913
Т
f
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_18971

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Т
Э
(__inference_conv2d_6_layer_call_fn_19394

inputs!
unknown: @
	unknown_0:@
identityИвStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ==@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_19032w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         ==@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         ?? : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         ?? 
 
_user_specified_nameinputs:%!

_user_specified_name19388:%!

_user_specified_name19390
н
_
C__inference_resizing_layer_call_and_return_conditional_losses_18617

inputs
identity\
resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"А   А   Ы
resize/ResizeBilinearResizeBilinearinputsresize/size:output:0*
T0*1
_output_shapes
:         АА*
half_pixel_centers(x
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*1
_output_shapes
:         АА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         АА:Y U
1
_output_shapes
:         АА
 
_user_specified_nameinputs
╬

є
B__inference_dense_3_layer_call_and_return_conditional_losses_19123

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
╖
№
C__inference_conv2d_5_layer_call_and_return_conditional_losses_19375

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ~~ *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ~~ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         ~~ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         ~~ S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         АА: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         АА
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Т
f
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_18991

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
ш
Ф
'__inference_dense_3_layer_call_fn_19545

inputs
unknown:@
	unknown_0:
identityИвStatefulPartitionedCall╫
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_19123o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs:%!

_user_specified_name19539:%!

_user_specified_name19541
▀
N
*__inference_sequential_layer_call_fn_18645
resizing_input
identity┬
PartitionedCallPartitionedCallresizing_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_18635j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         АА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         АА:a ]
1
_output_shapes
:         АА
(
_user_specified_nameresizing_input"╩L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╞
serving_default▓
W
sequential_inputC
"serving_default_sequential_input:0         АА;
dense_30
StatefulPartitionedCall:0         tensorflow/serving/predict:тв
ф
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer_with_weights-4
layer-10
layer-11
layer-12
layer_with_weights-5
layer-13
layer_with_weights-6
layer-14
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
─
layer-0
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses"
_tf_keras_sequential
─
!layer-0
"layer-1
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses"
_tf_keras_sequential
▌
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

/kernel
0bias
 1_jit_compiled_convolution_op"
_tf_keras_layer
е
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses"
_tf_keras_layer
▌
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses

>kernel
?bias
 @_jit_compiled_convolution_op"
_tf_keras_layer
е
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses"
_tf_keras_layer
▌
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses

Mkernel
Nbias
 O_jit_compiled_convolution_op"
_tf_keras_layer
е
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses"
_tf_keras_layer
▌
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses

\kernel
]bias
 ^_jit_compiled_convolution_op"
_tf_keras_layer
е
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses"
_tf_keras_layer
▌
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses

kkernel
lbias
 m_jit_compiled_convolution_op"
_tf_keras_layer
е
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses"
_tf_keras_layer
е
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses"
_tf_keras_layer
╜
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses
Аkernel
	Бbias"
_tf_keras_layer
├
В	variables
Гtrainable_variables
Дregularization_losses
Е	keras_api
Ж__call__
+З&call_and_return_all_conditional_losses
Иkernel
	Йbias"
_tf_keras_layer
К
/0
01
>2
?3
M4
N5
\6
]7
k8
l9
А10
Б11
И12
Й13"
trackable_list_wrapper
К
/0
01
>2
?3
M4
N5
\6
]7
k8
l9
А10
Б11
И12
Й13"
trackable_list_wrapper
 "
trackable_list_wrapper
╧
Кnon_trainable_variables
Лlayers
Мmetrics
 Нlayer_regularization_losses
Оlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
╧
Пtrace_0
Рtrace_12Ф
,__inference_sequential_3_layer_call_fn_19214
,__inference_sequential_3_layer_call_fn_19247╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zПtrace_0zРtrace_1
Е
Сtrace_0
Тtrace_12╩
G__inference_sequential_3_layer_call_and_return_conditional_losses_19130
G__inference_sequential_3_layer_call_and_return_conditional_losses_19177╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zСtrace_0zТtrace_1
╘B╤
 __inference__wrapped_model_18609sequential_input"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
°
	Уiter
Фbeta_1
Хbeta_2

Цdecay
Чlearning_rate/m╧0m╨>m╤?m╥Mm╙Nm╘\m╒]m╓km╫lm╪	Аm┘	Бm┌	Иm█	Йm▄/v▌0v▐>v▀?vрMvсNvт\vу]vфkvхlvц	Аvч	Бvш	Иvщ	Йvъ"
	optimizer
-
Шserving_default"
signature_map
л
Щ	variables
Ъtrainable_variables
Ыregularization_losses
Ь	keras_api
Э__call__
+Ю&call_and_return_all_conditional_losses"
_tf_keras_layer
л
Я	variables
аtrainable_variables
бregularization_losses
в	keras_api
г__call__
+д&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
еnon_trainable_variables
жlayers
зmetrics
 иlayer_regularization_losses
йlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
╦
кtrace_0
лtrace_12Р
*__inference_sequential_layer_call_fn_18640
*__inference_sequential_layer_call_fn_18645╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zкtrace_0zлtrace_1
Б
мtrace_0
нtrace_12╞
E__inference_sequential_layer_call_and_return_conditional_losses_18629
E__inference_sequential_layer_call_and_return_conditional_losses_18635╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zмtrace_0zнtrace_1
├
о	variables
пtrainable_variables
░regularization_losses
▒	keras_api
▓__call__
+│&call_and_return_all_conditional_losses
┤_random_generator"
_tf_keras_layer
├
╡	variables
╢trainable_variables
╖regularization_losses
╕	keras_api
╣__call__
+║&call_and_return_all_conditional_losses
╗_random_generator"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╝non_trainable_variables
╜layers
╛metrics
 ┐layer_regularization_losses
└layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
╧
┴trace_0
┬trace_12Ф
,__inference_sequential_1_layer_call_fn_18917
,__inference_sequential_1_layer_call_fn_18922╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┴trace_0z┬trace_1
Е
├trace_0
─trace_12╩
G__inference_sequential_1_layer_call_and_return_conditional_losses_18894
G__inference_sequential_1_layer_call_and_return_conditional_losses_18908╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z├trace_0z─trace_1
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
┼non_trainable_variables
╞layers
╟metrics
 ╚layer_regularization_losses
╔layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
ф
╩trace_02┼
(__inference_conv2d_5_layer_call_fn_19364Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╩trace_0
 
╦trace_02р
C__inference_conv2d_5_layer_call_and_return_conditional_losses_19375Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╦trace_0
):' 2conv2d_5/kernel
: 2conv2d_5/bias
к2зд
Ы▓Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╠non_trainable_variables
═layers
╬metrics
 ╧layer_regularization_losses
╨layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
ы
╤trace_02╠
/__inference_max_pooling2d_5_layer_call_fn_19380Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╤trace_0
Ж
╥trace_02ч
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_19385Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╥trace_0
.
>0
?1"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╙non_trainable_variables
╘layers
╒metrics
 ╓layer_regularization_losses
╫layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
ф
╪trace_02┼
(__inference_conv2d_6_layer_call_fn_19394Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╪trace_0
 
┘trace_02р
C__inference_conv2d_6_layer_call_and_return_conditional_losses_19405Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┘trace_0
):' @2conv2d_6/kernel
:@2conv2d_6/bias
к2зд
Ы▓Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
┌non_trainable_variables
█layers
▄metrics
 ▌layer_regularization_losses
▐layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
ы
▀trace_02╠
/__inference_max_pooling2d_6_layer_call_fn_19410Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z▀trace_0
Ж
рtrace_02ч
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_19415Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zрtrace_0
.
M0
N1"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
сnon_trainable_variables
тlayers
уmetrics
 фlayer_regularization_losses
хlayer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
ф
цtrace_02┼
(__inference_conv2d_7_layer_call_fn_19424Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zцtrace_0
 
чtrace_02р
C__inference_conv2d_7_layer_call_and_return_conditional_losses_19435Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zчtrace_0
):'@@2conv2d_7/kernel
:@2conv2d_7/bias
к2зд
Ы▓Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
шnon_trainable_variables
щlayers
ъmetrics
 ыlayer_regularization_losses
ьlayer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
ы
эtrace_02╠
/__inference_max_pooling2d_7_layer_call_fn_19440Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zэtrace_0
Ж
юtrace_02ч
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_19445Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zюtrace_0
.
\0
]1"
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
яnon_trainable_variables
Ёlayers
ёmetrics
 Єlayer_regularization_losses
єlayer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
ф
Їtrace_02┼
(__inference_conv2d_8_layer_call_fn_19454Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЇtrace_0
 
їtrace_02р
C__inference_conv2d_8_layer_call_and_return_conditional_losses_19465Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zїtrace_0
):'@@2conv2d_8/kernel
:@2conv2d_8/bias
к2зд
Ы▓Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Ўnon_trainable_variables
ўlayers
°metrics
 ∙layer_regularization_losses
·layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
ы
√trace_02╠
/__inference_max_pooling2d_8_layer_call_fn_19470Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z√trace_0
Ж
№trace_02ч
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_19475Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z№trace_0
.
k0
l1"
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
¤non_trainable_variables
■layers
 metrics
 Аlayer_regularization_losses
Бlayer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
ф
Вtrace_02┼
(__inference_conv2d_9_layer_call_fn_19484Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zВtrace_0
 
Гtrace_02р
C__inference_conv2d_9_layer_call_and_return_conditional_losses_19495Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zГtrace_0
):'@@2conv2d_9/kernel
:@2conv2d_9/bias
к2зд
Ы▓Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Дnon_trainable_variables
Еlayers
Жmetrics
 Зlayer_regularization_losses
Иlayer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
ы
Йtrace_02╠
/__inference_max_pooling2d_9_layer_call_fn_19500Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЙtrace_0
Ж
Кtrace_02ч
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_19505Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zКtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Лnon_trainable_variables
Мlayers
Нmetrics
 Оlayer_regularization_losses
Пlayer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
х
Рtrace_02╞
)__inference_flatten_1_layer_call_fn_19510Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zРtrace_0
А
Сtrace_02с
D__inference_flatten_1_layer_call_and_return_conditional_losses_19516Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zСtrace_0
0
А0
Б1"
trackable_list_wrapper
0
А0
Б1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Тnon_trainable_variables
Уlayers
Фmetrics
 Хlayer_regularization_losses
Цlayer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
у
Чtrace_02─
'__inference_dense_2_layer_call_fn_19525Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЧtrace_0
■
Шtrace_02▀
B__inference_dense_2_layer_call_and_return_conditional_losses_19536Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zШtrace_0
!:	А@2dense_2/kernel
:@2dense_2/bias
0
И0
Й1"
trackable_list_wrapper
0
И0
Й1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Щnon_trainable_variables
Ъlayers
Ыmetrics
 Ьlayer_regularization_losses
Эlayer_metrics
В	variables
Гtrainable_variables
Дregularization_losses
Ж__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
у
Юtrace_02─
'__inference_dense_3_layer_call_fn_19545Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЮtrace_0
■
Яtrace_02▀
B__inference_dense_3_layer_call_and_return_conditional_losses_19556Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЯtrace_0
 :@2dense_3/kernel
:2dense_3/bias
 "
trackable_list_wrapper
О
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14"
trackable_list_wrapper
0
а0
б1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
¤B·
,__inference_sequential_3_layer_call_fn_19214sequential_input"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
¤B·
,__inference_sequential_3_layer_call_fn_19247sequential_input"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ШBХ
G__inference_sequential_3_layer_call_and_return_conditional_losses_19130sequential_input"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ШBХ
G__inference_sequential_3_layer_call_and_return_conditional_losses_19177sequential_input"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
╙B╨
#__inference_signature_wrapper_19355sequential_input"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
вnon_trainable_variables
гlayers
дmetrics
 еlayer_regularization_losses
жlayer_metrics
Щ	variables
Ъtrainable_variables
Ыregularization_losses
Э__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses"
_generic_user_object
ф
зtrace_02┼
(__inference_resizing_layer_call_fn_19561Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zзtrace_0
 
иtrace_02р
C__inference_resizing_layer_call_and_return_conditional_losses_19567Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zиtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
йnon_trainable_variables
кlayers
лmetrics
 мlayer_regularization_losses
нlayer_metrics
Я	variables
аtrainable_variables
бregularization_losses
г__call__
+д&call_and_return_all_conditional_losses
'д"call_and_return_conditional_losses"
_generic_user_object
х
оtrace_02╞
)__inference_rescaling_layer_call_fn_19572Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zоtrace_0
А
пtrace_02с
D__inference_rescaling_layer_call_and_return_conditional_losses_19580Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zпtrace_0
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
∙BЎ
*__inference_sequential_layer_call_fn_18640resizing_input"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
∙BЎ
*__inference_sequential_layer_call_fn_18645resizing_input"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ФBС
E__inference_sequential_layer_call_and_return_conditional_losses_18629resizing_input"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ФBС
E__inference_sequential_layer_call_and_return_conditional_losses_18635resizing_input"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
░non_trainable_variables
▒layers
▓metrics
 │layer_regularization_losses
┤layer_metrics
о	variables
пtrainable_variables
░regularization_losses
▓__call__
+│&call_and_return_all_conditional_losses
'│"call_and_return_conditional_losses"
_generic_user_object
┴
╡trace_0
╢trace_12Ж
+__inference_random_flip_layer_call_fn_19587
+__inference_random_flip_layer_call_fn_19592й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╡trace_0z╢trace_1
ў
╖trace_0
╕trace_12╝
F__inference_random_flip_layer_call_and_return_conditional_losses_19703
F__inference_random_flip_layer_call_and_return_conditional_losses_19707й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╖trace_0z╕trace_1
/
╣
_generator"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
║non_trainable_variables
╗layers
╝metrics
 ╜layer_regularization_losses
╛layer_metrics
╡	variables
╢trainable_variables
╖regularization_losses
╣__call__
+║&call_and_return_all_conditional_losses
'║"call_and_return_conditional_losses"
_generic_user_object
╔
┐trace_0
└trace_12О
/__inference_random_rotation_layer_call_fn_19714
/__inference_random_rotation_layer_call_fn_19719й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┐trace_0z└trace_1
 
┴trace_0
┬trace_12─
J__inference_random_rotation_layer_call_and_return_conditional_losses_19837
J__inference_random_rotation_layer_call_and_return_conditional_losses_19841й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┴trace_0z┬trace_1
/
├
_generator"
_generic_user_object
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
■B√
,__inference_sequential_1_layer_call_fn_18917random_flip_input"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
■B√
,__inference_sequential_1_layer_call_fn_18922random_flip_input"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЩBЦ
G__inference_sequential_1_layer_call_and_return_conditional_losses_18894random_flip_input"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЩBЦ
G__inference_sequential_1_layer_call_and_return_conditional_losses_18908random_flip_input"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╥B╧
(__inference_conv2d_5_layer_call_fn_19364inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
эBъ
C__inference_conv2d_5_layer_call_and_return_conditional_losses_19375inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
┘B╓
/__inference_max_pooling2d_5_layer_call_fn_19380inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЇBё
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_19385inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╥B╧
(__inference_conv2d_6_layer_call_fn_19394inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
эBъ
C__inference_conv2d_6_layer_call_and_return_conditional_losses_19405inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
┘B╓
/__inference_max_pooling2d_6_layer_call_fn_19410inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЇBё
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_19415inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╥B╧
(__inference_conv2d_7_layer_call_fn_19424inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
эBъ
C__inference_conv2d_7_layer_call_and_return_conditional_losses_19435inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
┘B╓
/__inference_max_pooling2d_7_layer_call_fn_19440inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЇBё
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_19445inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╥B╧
(__inference_conv2d_8_layer_call_fn_19454inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
эBъ
C__inference_conv2d_8_layer_call_and_return_conditional_losses_19465inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
┘B╓
/__inference_max_pooling2d_8_layer_call_fn_19470inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЇBё
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_19475inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╥B╧
(__inference_conv2d_9_layer_call_fn_19484inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
эBъ
C__inference_conv2d_9_layer_call_and_return_conditional_losses_19495inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
┘B╓
/__inference_max_pooling2d_9_layer_call_fn_19500inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЇBё
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_19505inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╙B╨
)__inference_flatten_1_layer_call_fn_19510inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
юBы
D__inference_flatten_1_layer_call_and_return_conditional_losses_19516inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╤B╬
'__inference_dense_2_layer_call_fn_19525inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ьBщ
B__inference_dense_2_layer_call_and_return_conditional_losses_19536inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╤B╬
'__inference_dense_3_layer_call_fn_19545inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ьBщ
B__inference_dense_3_layer_call_and_return_conditional_losses_19556inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
R
─	variables
┼	keras_api

╞total

╟count"
_tf_keras_metric
c
╚	variables
╔	keras_api

╩total

╦count
╠
_fn_kwargs"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╥B╧
(__inference_resizing_layer_call_fn_19561inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
эBъ
C__inference_resizing_layer_call_and_return_conditional_losses_19567inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╙B╨
)__inference_rescaling_layer_call_fn_19572inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
юBы
D__inference_rescaling_layer_call_and_return_conditional_losses_19580inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
цBу
+__inference_random_flip_layer_call_fn_19587inputs"й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
цBу
+__inference_random_flip_layer_call_fn_19592inputs"й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
БB■
F__inference_random_flip_layer_call_and_return_conditional_losses_19703inputs"й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
БB■
F__inference_random_flip_layer_call_and_return_conditional_losses_19707inputs"й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
/
═
_state_var"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ъBч
/__inference_random_rotation_layer_call_fn_19714inputs"й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ъBч
/__inference_random_rotation_layer_call_fn_19719inputs"й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЕBВ
J__inference_random_rotation_layer_call_and_return_conditional_losses_19837inputs"й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЕBВ
J__inference_random_rotation_layer_call_and_return_conditional_losses_19841inputs"й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
/
╬
_state_var"
_generic_user_object
0
╞0
╟1"
trackable_list_wrapper
.
─	variables"
_generic_user_object
:  (2total
:  (2count
0
╩0
╦1"
trackable_list_wrapper
.
╚	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
 :	2random_flip/StateVar
$:"	2random_rotation/StateVar
.:, 2Adam/conv2d_5/kernel/m
 : 2Adam/conv2d_5/bias/m
.:, @2Adam/conv2d_6/kernel/m
 :@2Adam/conv2d_6/bias/m
.:,@@2Adam/conv2d_7/kernel/m
 :@2Adam/conv2d_7/bias/m
.:,@@2Adam/conv2d_8/kernel/m
 :@2Adam/conv2d_8/bias/m
.:,@@2Adam/conv2d_9/kernel/m
 :@2Adam/conv2d_9/bias/m
&:$	А@2Adam/dense_2/kernel/m
:@2Adam/dense_2/bias/m
%:#@2Adam/dense_3/kernel/m
:2Adam/dense_3/bias/m
.:, 2Adam/conv2d_5/kernel/v
 : 2Adam/conv2d_5/bias/v
.:, @2Adam/conv2d_6/kernel/v
 :@2Adam/conv2d_6/bias/v
.:,@@2Adam/conv2d_7/kernel/v
 :@2Adam/conv2d_7/bias/v
.:,@@2Adam/conv2d_8/kernel/v
 :@2Adam/conv2d_8/bias/v
.:,@@2Adam/conv2d_9/kernel/v
 :@2Adam/conv2d_9/bias/v
&:$	А@2Adam/dense_2/kernel/v
:@2Adam/dense_2/bias/v
%:#@2Adam/dense_3/kernel/v
:2Adam/dense_3/bias/v▒
 __inference__wrapped_model_18609М/0>?MN\]klАБИЙCв@
9в6
4К1
sequential_input         АА
к "1к.
,
dense_3!К
dense_3         ╝
C__inference_conv2d_5_layer_call_and_return_conditional_losses_19375u/09в6
/в,
*К'
inputs         АА
к "4в1
*К'
tensor_0         ~~ 
Ъ Ц
(__inference_conv2d_5_layer_call_fn_19364j/09в6
/в,
*К'
inputs         АА
к ")К&
unknown         ~~ ║
C__inference_conv2d_6_layer_call_and_return_conditional_losses_19405s>?7в4
-в*
(К%
inputs         ?? 
к "4в1
*К'
tensor_0         ==@
Ъ Ф
(__inference_conv2d_6_layer_call_fn_19394h>?7в4
-в*
(К%
inputs         ?? 
к ")К&
unknown         ==@║
C__inference_conv2d_7_layer_call_and_return_conditional_losses_19435sMN7в4
-в*
(К%
inputs         @
к "4в1
*К'
tensor_0         @
Ъ Ф
(__inference_conv2d_7_layer_call_fn_19424hMN7в4
-в*
(К%
inputs         @
к ")К&
unknown         @║
C__inference_conv2d_8_layer_call_and_return_conditional_losses_19465s\]7в4
-в*
(К%
inputs         @
к "4в1
*К'
tensor_0         @
Ъ Ф
(__inference_conv2d_8_layer_call_fn_19454h\]7в4
-в*
(К%
inputs         @
к ")К&
unknown         @║
C__inference_conv2d_9_layer_call_and_return_conditional_losses_19495skl7в4
-в*
(К%
inputs         @
к "4в1
*К'
tensor_0         @
Ъ Ф
(__inference_conv2d_9_layer_call_fn_19484hkl7в4
-в*
(К%
inputs         @
к ")К&
unknown         @м
B__inference_dense_2_layer_call_and_return_conditional_losses_19536fАБ0в-
&в#
!К
inputs         А
к ",в)
"К
tensor_0         @
Ъ Ж
'__inference_dense_2_layer_call_fn_19525[АБ0в-
&в#
!К
inputs         А
к "!К
unknown         @л
B__inference_dense_3_layer_call_and_return_conditional_losses_19556eИЙ/в,
%в"
 К
inputs         @
к ",в)
"К
tensor_0         
Ъ Е
'__inference_dense_3_layer_call_fn_19545ZИЙ/в,
%в"
 К
inputs         @
к "!К
unknown         ░
D__inference_flatten_1_layer_call_and_return_conditional_losses_19516h7в4
-в*
(К%
inputs         @
к "-в*
#К 
tensor_0         А
Ъ К
)__inference_flatten_1_layer_call_fn_19510]7в4
-в*
(К%
inputs         @
к ""К
unknown         АЇ
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_19385еRвO
HвE
CК@
inputs4                                    
к "OвL
EКB
tensor_04                                    
Ъ ╬
/__inference_max_pooling2d_5_layer_call_fn_19380ЪRвO
HвE
CК@
inputs4                                    
к "DКA
unknown4                                    Ї
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_19415еRвO
HвE
CК@
inputs4                                    
к "OвL
EКB
tensor_04                                    
Ъ ╬
/__inference_max_pooling2d_6_layer_call_fn_19410ЪRвO
HвE
CК@
inputs4                                    
к "DКA
unknown4                                    Ї
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_19445еRвO
HвE
CК@
inputs4                                    
к "OвL
EКB
tensor_04                                    
Ъ ╬
/__inference_max_pooling2d_7_layer_call_fn_19440ЪRвO
HвE
CК@
inputs4                                    
к "DКA
unknown4                                    Ї
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_19475еRвO
HвE
CК@
inputs4                                    
к "OвL
EКB
tensor_04                                    
Ъ ╬
/__inference_max_pooling2d_8_layer_call_fn_19470ЪRвO
HвE
CК@
inputs4                                    
к "DКA
unknown4                                    Ї
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_19505еRвO
HвE
CК@
inputs4                                    
к "OвL
EКB
tensor_04                                    
Ъ ╬
/__inference_max_pooling2d_9_layer_call_fn_19500ЪRвO
HвE
CК@
inputs4                                    
к "DКA
unknown4                                    ┼
F__inference_random_flip_layer_call_and_return_conditional_losses_19703{═=в:
3в0
*К'
inputs         АА
p
к "6в3
,К)
tensor_0         АА
Ъ ┴
F__inference_random_flip_layer_call_and_return_conditional_losses_19707w=в:
3в0
*К'
inputs         АА
p 
к "6в3
,К)
tensor_0         АА
Ъ Я
+__inference_random_flip_layer_call_fn_19587p═=в:
3в0
*К'
inputs         АА
p
к "+К(
unknown         ААЫ
+__inference_random_flip_layer_call_fn_19592l=в:
3в0
*К'
inputs         АА
p 
к "+К(
unknown         АА╔
J__inference_random_rotation_layer_call_and_return_conditional_losses_19837{╬=в:
3в0
*К'
inputs         АА
p
к "6в3
,К)
tensor_0         АА
Ъ ┼
J__inference_random_rotation_layer_call_and_return_conditional_losses_19841w=в:
3в0
*К'
inputs         АА
p 
к "6в3
,К)
tensor_0         АА
Ъ г
/__inference_random_rotation_layer_call_fn_19714p╬=в:
3в0
*К'
inputs         АА
p
к "+К(
unknown         ААЯ
/__inference_random_rotation_layer_call_fn_19719l=в:
3в0
*К'
inputs         АА
p 
к "+К(
unknown         АА╗
D__inference_rescaling_layer_call_and_return_conditional_losses_19580s9в6
/в,
*К'
inputs         АА
к "6в3
,К)
tensor_0         АА
Ъ Х
)__inference_rescaling_layer_call_fn_19572h9в6
/в,
*К'
inputs         АА
к "+К(
unknown         АА║
C__inference_resizing_layer_call_and_return_conditional_losses_19567s9в6
/в,
*К'
inputs         АА
к "6в3
,К)
tensor_0         АА
Ъ Ф
(__inference_resizing_layer_call_fn_19561h9в6
/в,
*К'
inputs         АА
к "+К(
unknown         АА╪
G__inference_sequential_1_layer_call_and_return_conditional_losses_18894М═╬LвI
Bв?
5К2
random_flip_input         АА
p

 
к "6в3
,К)
tensor_0         АА
Ъ ╥
G__inference_sequential_1_layer_call_and_return_conditional_losses_18908ЖLвI
Bв?
5К2
random_flip_input         АА
p 

 
к "6в3
,К)
tensor_0         АА
Ъ ▓
,__inference_sequential_1_layer_call_fn_18917Б═╬LвI
Bв?
5К2
random_flip_input         АА
p

 
к "+К(
unknown         ААл
,__inference_sequential_1_layer_call_fn_18922{LвI
Bв?
5К2
random_flip_input         АА
p 

 
к "+К(
unknown         АА▀
G__inference_sequential_3_layer_call_and_return_conditional_losses_19130У═╬/0>?MN\]klАБИЙKвH
Aв>
4К1
sequential_input         АА
p

 
к ",в)
"К
tensor_0         
Ъ █
G__inference_sequential_3_layer_call_and_return_conditional_losses_19177П/0>?MN\]klАБИЙKвH
Aв>
4К1
sequential_input         АА
p 

 
к ",в)
"К
tensor_0         
Ъ ╣
,__inference_sequential_3_layer_call_fn_19214И═╬/0>?MN\]klАБИЙKвH
Aв>
4К1
sequential_input         АА
p

 
к "!К
unknown         ╡
,__inference_sequential_3_layer_call_fn_19247Д/0>?MN\]klАБИЙKвH
Aв>
4К1
sequential_input         АА
p 

 
к "!К
unknown         ═
E__inference_sequential_layer_call_and_return_conditional_losses_18629ГIвF
?в<
2К/
resizing_input         АА
p

 
к "6в3
,К)
tensor_0         АА
Ъ ═
E__inference_sequential_layer_call_and_return_conditional_losses_18635ГIвF
?в<
2К/
resizing_input         АА
p 

 
к "6в3
,К)
tensor_0         АА
Ъ ж
*__inference_sequential_layer_call_fn_18640xIвF
?в<
2К/
resizing_input         АА
p

 
к "+К(
unknown         ААж
*__inference_sequential_layer_call_fn_18645xIвF
?в<
2К/
resizing_input         АА
p 

 
к "+К(
unknown         АА╚
#__inference_signature_wrapper_19355а/0>?MN\]klАБИЙWвT
в 
MкJ
H
sequential_input4К1
sequential_input         АА"1к.
,
dense_3!К
dense_3         