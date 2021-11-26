package edge

type Allocator interface {
	Reset(map[int]*Source)
	Allocate()
}

type EvenAllocator struct {
	Allocator
	sources map[int]*Source
}

func (a *EvenAllocator) Reset(sources map[int]*Source) {
	a.sources = sources
}

func (a *EvenAllocator) Allocate() {
	for _, source := range a.sources {
		source.weight = 1
	}
}

type AccuracyBasedAllocator struct {
	Allocator
	sources map[int]*Source
}

func (a *AccuracyBasedAllocator) Reset(sources map[int]*Source) {
	a.sources = sources
}

func (a *AccuracyBasedAllocator) Allocate() {
	for _, source := range a.sources {
		source.m.RLock()
		source.weight = 1 - source.profiles[len(source.profiles)-1].accuracy
		source.m.RUnlock()
	}
}
