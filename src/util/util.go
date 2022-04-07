package util

import "fmt"

func GCD(l []int) int {
	if len(l) == 1 {
		return l[0]
	}
	gcd := func(a int, b int) int {
		for b != 0 {
			t := b
			b = a % b
			a = t
		}
		return a
	}
	var r int
	for i := 0; i < len(l)-1; i++ {
		r = gcd(l[i], l[i+1])
	}
	return r
}

func Reduce(m map[int]int) int {
	l := make([]int, 0)
	for _, v := range m {
		l = append(l, v)
	}
	gcd := GCD(l)
	for k, v := range m {
		m[k] = v / gcd
	}
	return gcd
}

func GenerateSamplePosition(sampleCount int, sampleInterval int, offset int) []int {
	sampleWindow, count := make([]int, sampleCount), 0
	for count < sampleInterval {
		for i := 0; i < sampleCount; i++ {
			sampleWindow[i]++
			count++
			if count == sampleInterval {
				break
			}
		}
	}
	samplePosition := []int{offset}
	for i := 0; i < sampleCount-1; i++ {
		samplePosition = append(samplePosition, samplePosition[len(samplePosition)-1]+sampleWindow[i])
	}
	return samplePosition
}

func Exist(l []int, target int) bool {
	for i := 0; i < len(l); i++ {
		if l[i] == target {
			return true
		}
	}
	return false
}

func SourceGetFrameName(index int) string {
	return fmt.Sprintf("%06d.jpg", index)
}

func EdgeGetFrameName(source int, index int) string {
	return fmt.Sprintf("%d-%06d.jpg", source, index)
}

func EdgeGetModelName(source int, version int) string {
	return fmt.Sprintf("%d-%d.pth", source, version)
}

func CloudGetFrameName(edge int, source int, index int) string {
	return fmt.Sprintf("%d-%d-%06d.jpg", edge, source, index)
}

func CloudGetModelName(edge int, source int, version int) string {
	return fmt.Sprintf("%d-%d-%d.pth", edge, source, version)
}
