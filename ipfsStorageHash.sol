// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract UserHashMapping {
    // Mapping from user ID to hash value (as a string)
    mapping(string => string) private userHashes;

    // Function to store a user ID and its corresponding hash
    function storeUserHash(string memory userId, string memory hashValue) public {
        userHashes[userId] = hashValue;
    }

    // Function to retrieve the hash for a given user ID
    function getUserHash(string memory userId) public view returns (string memory) {
        return userHashes[userId];
    }
}
