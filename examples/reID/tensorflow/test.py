  # -*- coding: utf-8 -*
import numpy as np
def layer(op):
	def layer_decorated(self, *args, **kwargs):
		name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
		# Figure out the layer inputs.
		if len(self.terminals) == 0:
			raise RuntimeError('No input variables found for layer %s.' % name)
		elif len(self.terminals) == 1:
			layer_input = self.terminals[0]
		else:
			layer_input = list(self.terminals)
		# Perform the operation and get the output.
		layer_output = op(self, layer_input, *args, **kwargs)
		# Add to layer LUT.
		self.layers[name] = layer_output
		# This output is now the input for the next layer.
		self.feed(layer_output)
		# Return self for chained calls.
		return self

	return layer_decorated

#注意这个位置的链式调用，仔细分析一下其用法:
#feed()的返回值是对象实例self，而conv()的返回值是tf.nn结果。
#But!!!conv被layer装饰器wrapper了一下，在wrapper中最后几行代码：
#layer_output = op(self, layer_input, *args, **kwargs)
#self.layers[name] = layer_output
#self.feed(layer_output)
#return self
#能够看到conv的结果被放到了layer中，并且给到了feed函数，而feed函数又会把这个结果方到inputs里，这时候返回self。
#由于返回的是self，那么可以链式进行下一次调用。在wrapper的开始位置又会从inputs中取出layer_input，所以我们在看到下面的conv()调用中没有输入input，但实际上是在wrapper的调#用中给出了。


class BaseNetwork(object):
	def __init__(self, inputs, trainable=True):
		# The input nodes for this network
		self.inputs = inputs
		# The current list of terminal nodes
		self.terminals = []
		# Mapping from layer names to layers
		self.layers = dict(inputs)
		# If true, the resulting variables are set as trainable
		self.trainable = trainable
		self.struct = 0
		self.sum = 0
		self.setup()
	def get_unique_name(self, prefix):
		"""Returns an index-suffixed unique name for the given prefix.
		This is used for auto-generating layer names based on the type-prefix.
		"""
		ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
		return '%s_%d' % (prefix, ident)
	def setup(self):
		"""Construct the network. """
		raise NotImplementedError('Must be implemented by the subclass.')
	def feed(self, *args):
		"""Set the input(s) for the next operation by replacing the terminal nodes.
		The arguments can be either layer names or the actual layers.
		"""
		print("args: ", args)
		assert len(args) != 0
		self.terminals = []
		for fed_layer in args:#
			try:
				is_str = isinstance(fed_layer, basestring)
			except NameError:
				is_str = isinstance(fed_layer, str)
			if is_str:
				try:
					fed_layer = self.layers[fed_layer]
					print(fed_layer)
				except KeyError:
					raise KeyError('Unknown layer name fed: %s' % fed_layer)
			self.terminals.append(fed_layer)
		return self
	@layer
	def test_axpb(self,inputs, name):
		out = np.sum(inputs)
		print(out)
		outmean = inputs // out
		print(outmean)
		output ={name: outmean}
		return output
	@layer
	def test_asub(self, inputs, name):
		out = np.std(inputs, axis=1)
		print(out)
		output ={name: out}
		return output

class PNet(BaseNetwork):
	def setup(self):
		print((self.feed('data')
		.test_asub(name='123')))
#		.test_axpb(name='qed')


def max():
    print("i was great!")
    return self


if __name__=='__main__':
	inputs = np.array([[3.2, 2.5],[2.4, 5.5], [6.5, 3.3]])
	pnet = PNet({'data':inputs})
	max().max().max()
