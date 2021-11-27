package util

import (
	"math"
	"sort"
)

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

type unit struct {
	id      int
	current int
	target  float64
}

type units []unit

func (u units) Len() int {
	return len(u)
}

func (u units) Less(i, j int) bool {
	return float64(u[i].current)-u[i].target < float64(u[j].current)-u[j].target
}

func (u units) Swap(i, j int) {
	u[i], u[j] = u[j], u[i]
}

func Allocate(target map[int]float64, current map[int]int) {
	l := make(units, 0)
	if len(current) == 1 {
		return
	}
	for id := range target {
		l = append(l, unit{
			id:      id,
			current: current[id],
			target:  target[id],
		})
	}
	sort.Sort(l)
	for {
		victim := len(l) - 1
		for victim > 0 && l[victim].current == 1 {
			victim--
		}
		if victim <= 0 {
			return
		}
		p1 := math.Abs(float64(l[0].current)-l[0].target) + math.Abs(float64(l[victim].current)-l[victim].target)
		p2 := math.Abs(float64(l[0].current+1)-l[0].target) + math.Abs(float64(l[victim].current-1)-l[victim].target)
		if p1 <= p2 {
			return
		}
		l[0].current++
		l[victim].current--
		current[l[0].id]++
		current[l[victim].id]--
		sort.Sort(l)
	}
}
