    const auto placement_error = params.route_b_placement_error();
    if (!placement_error.empty()) {
        throw std::invalid_argument(placement_error);
    }
