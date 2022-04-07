package cloud

type ResourceMonitor interface {
	GetResource() float64
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
