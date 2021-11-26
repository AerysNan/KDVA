package util

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
