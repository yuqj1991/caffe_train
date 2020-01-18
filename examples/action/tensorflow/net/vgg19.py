from net import network as Network

class VGG19_net(Network):
	def setup(self):
		(self.feed('data')
					.conv(3, 3, 64, 1, 1, name = 'conv1_1')
					.conv(3, 3, 64, 1, 1, name = 'conv1_2')
					.max_pool(3, 3, 2, 2, name='pool1')
					.conv(3, 3, 128, 1, 1, name = 'conv2_1')
					.conv(3, 3, 128, 1, 1, name = 'conv2_2')
					.max_pool(3, 3, 2, 2, name = 'pool2')
					.conv(3, 3, 256, 1, 1, name = 'conv3_1')
					.conv(3, 3, 256, 1, 1, name = 'conv3_2')
					.conv(3, 3, 256, 1, 1, name = 'conv3_3')
					.conv(3, 3, 256, 1, 1, name = 'conv3_4')
					.max_pool(3, 3, 2, 2, name = 'pool3')
					.conv(3, 3, 512, 1, 1, name = 'conv4_1')
					.conv(3, 3, 512, 1, 1, name = 'conv4_2')
					.conv(3, 3, 512, 1, 1, name = 'conv4_3')
					.conv(3, 3, 512, 1, 1, name = 'conv4_4')
					.max_pool(3, 3, 2, 2, name = 'pool4')
					.conv(3, 3, 512, 1, 1, name = 'conv5_1')
					.conv(3, 3, 512, 1, 1, name = 'conv5_2')
					.conv(3, 3, 512, 1, 1, name = 'conv5_3')
					.conv(3, 3, 512, 1, 1, name = 'conv5_4')
					.max_pool(3, 3, 2, 2, name = 'pool5')
					.fc(4096, name='cls1_fc0')
					.fc(4096, name='cls1_fc1')
					.fc(1024, name='cls1_fc2') # final endpoint
					.sofemax(name='cls1_softmax')
		)

