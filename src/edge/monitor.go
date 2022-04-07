package edge

type ResourceMonitor interface {
	GetResource() float64
}

type MockComputationMonitor struct {
	ResourceMonitor
	resource float64
}

func NewMockComputationMonitor(resource float64) *MockComputationMonitor {
	return &MockComputationMonitor{
		resource: resource,
	}
}

func (m *MockComputationMonitor) GetResource() float64 {
	return m.resource
}

type MockNetworkMonitor struct {
	ResourceMonitor
	resource float64
}

func NewMockNetworkMonitor(resource float64) *MockNetworkMonitor {
	return &MockNetworkMonitor{
		resource: resource,
	}
}

func (m *MockNetworkMonitor) GetResource() float64 {
	return m.resource
}
